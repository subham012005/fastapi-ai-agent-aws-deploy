from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import SystemMessage , ToolMessage , HumanMessage, AIMessage    
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# LLM
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


class SessionMemory:
    def __init__(self):
        self.sessions: Dict[str, List[str]] = {}

    def add(self, session_id: str, message: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(message)

    def get(self, session_id: str) -> List[str]:
        return self.sessions.get(session_id, [])

memory = SessionMemory()

policy_content = """
Company Policy Document:
First500Days operates under Manalang Ventures LLP, which provides a platform for communities to make themselves
discoverable and engage First500Days people (Service). This page is used to inform website visitors regarding our
policies with the collection, use, and disclosure of Personal Information if anyone decided to use our Service.

If you choose to use our Service, then you agree to the collection and use of information in relation with this policy.

The Personal Information that we collect are used for providing and improving the service. We will not use or share your information with anyone except as described in this Privacy Policy.

The terms used in this Privacy Policy have the same meanings as in our Terms and Conditions,
which is accessible at Website URL, unless otherwise defined in this Privacy Policy. By signing up, you expressly
consent to receive non-marketing and marketing text messages and emails from First500Days and others texting and
emailing on its behalf, at the telephone number(s) and emails that you provide.

You may opt-out of these communications at any time.

Information Collection and Use
For a better experience while using our Service, we may require you to provide us with certain personally
identifiable information, including but not limited to your name, email ID, phone number, work experience, city and
education background. The information that we collect will be used to contact or identify you.

Log Data
We want to inform you that whenever you visit our Service, we collect information that your browser sends to us
that is called Log Data. This Log Data may include information such as your computer’s Internet Protocol (“IP”)
address, browser version, pages of our Service that you visit, the time and date of your visit, the time spent on
those pages, and other statistics.

Cookies
Cookies are files with small amount of data that is commonly used an anonymous unique identifier. These are sent to
your browser from the website that you visit and are stored on your computer’s hard drive.

Our website uses these “cookies” to collection information and to improve our Service. You have the option to
either accept or refuse these cookies, and know when a cookie is being sent to your computer. If you choose to
refuse our cookies, you may not be able to use some portions of our Service

Service Providers
We may employ third-party companies and individuals due to the following reasons:

To facilitate our Service;

To provide the Service on our behalf;

To perform Service-related services; or

To assist us in analyzing how our Service is used.

We want to inform our Service users that these third parties have access to your Personal Information. The reason
is to perform the tasks assigned to them on our behalf. However, they are obligated not to disclose or use the
information for any other purpose.

Security
We value your trust in providing us your Personal Information, thus we are striving to use commercially acceptable
means of protecting it. But remember that no method of transmission over the internet, or method of electronic
storage is 100% secure and reliable, and we cannot guarantee its absolute security.

Refund Policy
This company don’t support any kind of refund of payment as they are providing digital products and services.
Refunds are acceptable only If there is any explicit claim of refund over email while making payment.

Children’s Privacy
Our Services do not address anyone under the age of 13. We do not knowingly collect personal identifiable
information from children under 13. In the case we discover that a child under 13 has provided us with personal
information, we immediately delete this from our servers. If you are a parent or guardian and you are aware that
your child has provided us with personal information, please contact us so that we will be able to do necessary
actions.

Changes to This Privacy Policy

We may update our Privacy Policy from time to time. Thus, we advise you to review this page periodically for any
changes. We will notify you of any changes by posting the new Privacy Policy on this page. These changes are
effective immediately, after they are posted on this page.

Contact Us
If you have any questions or suggestions about our Privacy Policy, do not hesitate to contact us.
"""


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

@tool
def retrieve_documents(query: str) -> str:
    """
    Retrieves and summarizes information from the internal company policy document to answer user queries.
    """
    prompt = f"""
 Internal Policy Document Reference
You are acting as a precise information retrieval system. Use the provided policy content below to answer the user's question accurately.

**Policy Content:**
{policy_content}

**Instructions:**
- Provide a concise, direct answer based ONLY on the policy above.
- If the information is not present, respond exactly with: "no relevant information present please reach out company directly".
- Maintain a professional and objective tone.

**User Question:** 
{query}
"""
    return llm.invoke(prompt).content.strip()


tools = [retrieve_documents]


agent = create_agent(
    tools=tools,
    middleware=[handle_tool_errors],
    model=llm,
    system_prompt= SystemMessage(
        content=(
            "You are the First500Days Internal Policy Assistant. Your primary role is to help employees navigate "
            "company policies using the provided tools.\n\n"
            "GUIDELINES:\n"
            "1. For policy-related questions, ALWAYS use the `retrieve_documents` tool.\n"
            "2. For general greetings or casual conversation, respond politely and professionally in a formal tone.\n"
            "3. If a question is outside the scope of company policies and cannot be answered by your general knowledge "
            "in a formal capacity, state that you do not have that information.\n"
            "4. Always maintain a helpful, professional, and formal demeanor."
        )
    ),
    checkpointer=InMemorySaver()
)

def run_query(agent, query: str, session_id: str = "1") -> str:
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": session_id}},
    )
    return response["messages"][-1].content

def run():
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        answer = run_query(agent, user_input)
        print(answer)
        
if __name__ == "__main__":
    run()
