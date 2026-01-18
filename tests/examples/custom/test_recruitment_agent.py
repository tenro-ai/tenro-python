# Copyright 2026 Tenro.ai
# SPDX-License-Identifier: Apache-2.0

"""Advanced example: Recruiting screening agent.

Tests an agent that screens candidates and schedules interviews.
"""

from tenro import Construct, Provider, link_agent, link_llm, link_tool
from tenro.simulate import tool

# APPLICATION CODE


@link_tool("fetch_resume")
def fetch_resume(candidate_id: str) -> dict:
    """Fetch candidate resume from ATS."""
    return {"name": "Jane Doe", "experience": "5 years Python", "skills": ["Python", "ML"]}


@link_tool("get_job_requirements")
def get_job_requirements(job_id: str) -> dict:
    """Get job posting requirements."""
    return {"title": "Senior ML Engineer", "required_skills": ["Python", "ML", "AWS"]}


@link_tool("schedule_interview")
def schedule_interview(candidate_id: str, interviewer_email: str) -> dict:
    """Schedule an interview via calendar API."""
    return {"scheduled": True, "time": "2025-01-02 10:00 AM"}


@link_tool("send_rejection_email")
def send_rejection_email(candidate_id: str, reason: str) -> bool:
    """Send a polite rejection email."""
    return True


@link_llm(Provider.OPENAI)
def evaluate_candidate(resume: dict, requirements: dict) -> dict:
    """Evaluate if candidate matches job requirements."""
    import json

    import openai

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Resume: {resume}\nRequirements: {requirements}\n\n"
                    "Respond with JSON: "
                    '{{"score": 1-10, "decision": "interview"|"reject", "notes": "..."}}'
                ),
            }
        ],
    )
    # Parse the JSON response from the LLM
    return json.loads(response.choices[0].message.content)


@link_agent("RecruitingScreeningAgent")
class RecruitingScreeningAgent:
    """Agent that screens candidates and takes appropriate action."""

    def run(self, candidate_id: str, job_id: str) -> dict:
        """Run the recruiting screening agent."""
        resume = fetch_resume(candidate_id)
        requirements = get_job_requirements(job_id)
        evaluation = evaluate_candidate(resume, requirements)

        if evaluation["decision"] == "interview":
            result = schedule_interview(candidate_id, "hiring-manager@company.com")
            return {"action": "interview_scheduled", "evaluation": evaluation, **result}
        else:
            send_rejection_email(candidate_id, evaluation["notes"])
            return {"action": "rejected", "evaluation": evaluation}


# TESTS


def test_recruiting_agent_schedules_qualified_candidate(construct: Construct):
    """Test that agent schedules interview for qualified candidates."""
    # Control what tools and LLMs return
    construct.simulate_tool(
        fetch_resume,
        result={"name": "Jane Doe", "experience": "5 years", "skills": ["Python", "ML", "AWS"]},
    )
    construct.simulate_tool(
        get_job_requirements,
        result={"title": "ML Engineer", "required_skills": ["Python", "ML"]},
    )
    construct.simulate_tool(
        schedule_interview,
        result={"scheduled": True, "time": "2025-01-02 10:00 AM"},
    )
    # Simulate LLM returning JSON with interview decision
    construct.simulate_llm(
        Provider.OPENAI,
        response='{"score": 9, "decision": "interview", "notes": "Strong candidate"}',
    )

    # Run the agent
    RecruitingScreeningAgent().run("candidate-123", "job-ml-engineer")

    # Verify correct path taken
    tool.verify_many(fetch_resume, count=1)
    tool.verify_many(get_job_requirements, count=1)
    tool.verify_many(schedule_interview, count=1)
    tool.verify_never(send_rejection_email)


def test_recruiting_agent_rejects_unqualified_candidate(construct: Construct):
    """Test that agent rejects candidates who don't meet requirements."""
    # Simulate unqualified candidate
    construct.simulate_tool(
        fetch_resume,
        result={"name": "John Doe", "experience": "1 year", "skills": ["JavaScript"]},
    )
    construct.simulate_tool(
        get_job_requirements,
        result={"title": "Senior ML Engineer", "required_skills": ["Python", "ML", "5+ years"]},
    )
    construct.simulate_tool(send_rejection_email, result=True)
    # Simulate LLM returning JSON with reject decision
    construct.simulate_llm(
        Provider.OPENAI,
        response='{"score": 3, "decision": "reject", "notes": "Lacks Python/ML experience"}',
    )

    # Run the agent
    RecruitingScreeningAgent().run("candidate-456", "job-senior-ml")

    # Verify rejection path
    tool.verify_many(fetch_resume, count=1)
    tool.verify_many(send_rejection_email, count=1)
    tool.verify_never(schedule_interview)
