"""Tool definitions and LangChain agent assembly.

build_agent_executor() takes already-constructed clients (engine, Qdrant
client, embedder, session factory) and settings, and returns the assembled
AgentExecutor — it does no client construction itself, so it has no import-
time or hidden side effects. It's called once from api.py's lifespan.
"""

from langchain.agents import AgentExecutor, create_sql_agent
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Qdrant
from langchain_groq import ChatGroq

from .config import Settings
from .models import Property

SYSTEM_PROMPT = """You are a specialized real-estate assistant.
Your goal is to answer questions about properties using the tools provided.
You can use the 'structured_property_search' tool for facts like price, location, and room counts.
You can use the 'unstructured_property_search' tool for more general questions about descriptions, neighborhoods, or report details.

If the user asks a question that is NOT related to real estate, properties, or your other tools,
and it is NOT a simple greeting (like 'hello'), you MUST respond with this exact sentence:
'I am a property information helper, I can't help you with this information.'

Do not answer general knowledge questions, write code, or discuss other topics.
"""


def build_agent_executor(
    *, engine, qdrant_client, embedder, session_factory, settings: Settings
) -> AgentExecutor:
    llm = ChatGroq(model=settings.LLM_MODEL_NAME, temperature=0, api_key=settings.GROQ_API_KEY)

    db = SQLDatabase(engine, include_tables=["properties"])
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_agent = create_sql_agent(llm=llm, toolkit=sql_toolkit, agent_type="openai-tools", verbose=True)
    sql_search_tool = Tool(
        name="structured_property_search",
        func=sql_agent.invoke,
        description="Use to query the 'properties' table for properties based on price, location, rooms, etc.",
    )

    vector_store = Qdrant(
        client=qdrant_client, collection_name=settings.QDRANT_VECTOR_COLLECTION, embeddings=embedder
    )
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_store.as_retriever()
    )
    rag_search_tool = Tool(
        name="unstructured_property_search",
        func=rag_chain.invoke,
        description="Use to search property descriptions for semantic info like 'family-friendly' or 'good view'.",
    )

    @tool
    def generate_property_report(
        location: str = None, min_price: float = None, max_price: float = None, min_rooms: int = None
    ) -> str:
        """
        Generates a summary report of properties matching search criteria.
        Use this when the user asks for a 'report', 'summary', or 'list' of properties.
        Arguments:
            location (str): The city or area to search in.
            min_price (float): The minimum price.
            max_price (float): The maximum price.
            min_rooms (int): The minimum number of rooms.
        """
        print("--- Report Generation Agent Triggered ---")
        db = session_factory()
        try:
            # Build the query dynamically based on provided arguments
            query = db.query(Property)
            filters = []

            if location:
                filters.append(Property.location.ilike(f"%{location}%"))
            if min_price is not None:
                filters.append(Property.price >= min_price)
            if max_price is not None:
                filters.append(Property.price <= max_price)
            if min_rooms is not None:
                filters.append(Property.rooms >= min_rooms)

            if filters:
                query = query.filter(*filters)

            properties = query.all()

            if not properties:
                return "I found no properties matching those criteria."

            # --- Create a Markdown report string ---
            report = "# Property Report\n\n"
            report += f"I found {len(properties)} properties matching your criteria.\n\n---\n\n"

            for prop in properties:
                price_str = f"₹{prop.price:,.0f}" if prop.price is not None else "N/A"
                rooms_str = str(prop.rooms) if prop.rooms is not None else "N/A"
                bath_str = str(prop.bathrooms) if prop.bathrooms is not None else "N/A"
                if prop.description:
                    desc_str = prop.description[:150] + ("..." if len(prop.description) > 150 else "")
                else:
                    desc_str = "N/A"

                report += f"### {prop.title or 'Untitled property'}\n"
                report += f"* **Location:** {prop.location or 'N/A'}\n"
                report += f"* **Price:** {price_str}\n"
                report += f"* **Rooms:** {rooms_str}\n"
                report += f"* **Bathrooms:** {bath_str}\n"
                report += f"* **Description:** {desc_str}\n\n"

            print(f"Generated report with {len(properties)} properties.")
            return report

        except Exception as e:
            print(f"Error generating report: {e}")
            return "Sorry, I was unable to generate the report due to an error."
        finally:
            db.close()

    # --- Assemble Agent ---
    tools = [sql_search_tool, rag_search_tool, generate_property_report]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm_with_tools = llm.bind_tools(tools)
    main_agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    return AgentExecutor(agent=main_agent, tools=tools, verbose=True)
