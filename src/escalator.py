import json

from src.config import (
    CONFIDENCE_THRESHOLD,
    SENSITIVE_TOPICS
)


class EscalationManager:
    """
    Handles escalation decisions and
    creates human handoff summaries.
    """


    def should_escalate(
        self,
        user_query,
        retrieved_documents
    ):
        """
        Check whether the conversation
        requires human support.
        """

        query_lower = user_query.lower()


        # Check sensitive issues
        for topic in SENSITIVE_TOPICS:
            if topic in query_lower:
                return True, (
                    "Sensitive issue detected "
                    f"({topic})"
                )


        # No documents found
        if not retrieved_documents:
            return True, (
                "No relevant knowledge base "
                "documents found"
            )


        # Check confidence score
        best_score = max(
            doc["score"]
            for doc in retrieved_documents
        )

        if best_score < CONFIDENCE_THRESHOLD:
            return True, (
                "Low retrieval confidence"
            )


        return False, "No escalation required"


    def generate_handoff_summary(
        self,
        persona,
        user_query,
        conversation_history,
        retrieved_documents
    ):
        """
        Generate structured handoff
        information for a human agent.
        """

        handoff = {

            "persona": persona,

            "issue_summary": user_query,

            "conversation_history":
                conversation_history,

            "documents_used": [
                doc["source"]
                for doc in retrieved_documents
            ],

            "actions_attempted": [
                "AI knowledge base retrieval",
                "Automated support response"
            ],

            "recommended_next_steps":
                "Human support agent should "
                "review the issue and continue "
                "troubleshooting."
        }


        return json.dumps(
            handoff,
            indent=4
        )