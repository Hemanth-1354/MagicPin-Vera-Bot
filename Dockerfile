FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py conversation_handlers.py ./

ENV PORT=8080
ENV GOOGLE_API_KEY=""
ENV ANTHROPIC_API_KEY=""
ENV TEAM_NAME="Vera-Builder"
ENV TEAM_MEMBERS="Candidate"
ENV CONTACT_EMAIL="candidate@example.com"

EXPOSE 8080

CMD ["uvicorn", "bot:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
