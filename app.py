import streamlit as st

from src.classifier import classify_customer_persona
from src.rag_pipeline import (
    DocumentLoader,
    VectorStore
)
from src.generator import ResponseGenerator
from src.escalator import EscalationManager


# ----------------------------
# Streamlit Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Persona Adaptive Support Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Persona-Adaptive Customer Support Agent")

st.write("""
This AI assistant detects customer personas,
retrieves knowledge from support documents,
and provides personalized customer support.
""")


# ----------------------------
# Initialize Components
# ----------------------------

document_loader = DocumentLoader()

vector_store = VectorStore()

response_generator = ResponseGenerator()

escalation_manager = EscalationManager()


# ----------------------------
# Knowledge Base Ingestion
# ----------------------------

if "knowledge_loaded" not in st.session_state:

    with st.spinner("Loading knowledge base..."):

        documents = document_loader.load_documents()

        chunks = document_loader.split_documents(
            documents
        )

        vector_store.store_documents(
            chunks
        )

        st.session_state.knowledge_loaded = True

    st.success("Knowledge Base Loaded Successfully!")


# ----------------------------
# Conversation History
# ----------------------------

if "history" not in st.session_state:
    st.session_state.history = []


# ----------------------------
# User Input
# ----------------------------

user_message = st.text_area(
    "Enter your support question:",
    height=100
)


if st.button("Submit"):

    if user_message.strip():

        # Store user message
        st.session_state.history.append(
            f"User: {user_message}"
        )

        # Persona Detection
        persona_result = classify_customer_persona(
            user_message
        )

        persona = persona_result["persona"]


        # Retrieve relevant documents
        retrieved_docs = vector_store.retrieve_documents(
            user_message
        )


        # Check escalation
        escalate, reason = (
            escalation_manager.should_escalate(
                user_message,
                retrieved_docs
            )
        )


        st.subheader("Detected Persona")
        st.write(persona)


        st.subheader("Retrieved Sources")

        if retrieved_docs:

            for doc in retrieved_docs:

                st.write(
                    f"📄 {doc['source']} "
                    f"(Score: {doc['score']:.2f})"
                )

        else:
            st.write("No relevant documents found.")


        # Escalation Flow
        if escalate:

            st.error(
                f"Escalated to Human Support: {reason}"
            )

            handoff = (
                escalation_manager
                .generate_handoff_summary(
                    persona,
                    user_message,
                    st.session_state.history,
                    retrieved_docs
                )
            )

            st.subheader(
                "Human Handoff Summary"
            )

            st.json(handoff)


        # AI Response Flow
        else:

            response = (
                response_generator.generate_response(
                    user_message,
                    persona,
                    retrieved_docs
                )
            )

            st.subheader(
                "AI Support Response"
            )

            st.write(response)


            st.session_state.history.append(
                f"AI: {response}"
            )


    else:
        st.warning(
            "Please enter a message."
        )