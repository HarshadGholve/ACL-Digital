feedback_storage = []


def store_feedback(question: str, answer: str, feedback: str):
    feedback_storage.append({"question": question, "answer": answer, "feedback": feedback})
    return "Feedback stored successfully!"
