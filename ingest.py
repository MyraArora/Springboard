import os
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# 1. This line reads the secret .env file and loads your API key into memory
load_dotenv()

# 2. This starts the OpenAI client using the key it just loaded
client = OpenAI()

# --- Schema Definitions ---
class FieldCategory(str, Enum):
    RESEARCH = "Research"
    DEVELOPMENT = "Development/Coding"
    WRITING = "Writing/Humanities"
    FINANCE = "Finance/Economics"
    PITCHING = "Pitching/Business"
    SOCIAL_IMPACT = "Social Impact/Leadership"
    OTHER = "Other"

class EligibilityCriteria(BaseModel):
    allowed_countries: List[str] = Field(description="Countries eligible. Use ['Global'] if open to all.")
    min_age: Optional[int] = Field(None, description="Minimum age allowed, if specified.")
    max_age: Optional[int] = Field(None, description="Maximum age allowed, if specified.")
    grade_levels: List[str] = Field(description="Target grade levels, e.g., ['9th', '10th', '11th', '12th'].")
    language: str = Field(default="English", description="Primary language required for submission.")

class SubmissionRequirements(BaseModel):
    has_video_pitch: bool = Field(description="True if a video pitch/presentation is required.")
    requires_code: bool = Field(description="True if writing code is mandatory.")
    requires_paper: bool = Field(description="True if a research paper or essay is mandatory.")
    deliverables_description: str = Field(description="A concise summary of what needs to be turned in.")

class EvaluationCriteria(BaseModel):
    innovation_weight: int = Field(description="Score from 0 to 10 indicating importance of creativity/novelty.")
    technical_weight: int = Field(description="Score from 0 to 10 indicating importance of engineering/coding.")
    research_weight: int = Field(description="Score from 0 to 10 indicating importance of academic/methodological rigor.")
    presentation_weight: int = Field(description="Score from 0 to 10 indicating importance of business pitch/communication.")

class Opportunity(BaseModel):
    name: str = Field(description="The formal name of the competition or program.")
    website_link: str = Field(description="The primary URL of the opportunity.")
    description: str = Field(description="A 2-3 sentence overview of what this competition/program is.")
    eligibility: EligibilityCriteria
    fields: List[FieldCategory] = Field(description="Categories that this opportunity falls under.")
    submission: SubmissionRequirements
    evaluation: EvaluationCriteria
    estimated_prestige: int = Field(description="A score from 1 to 10 estimating the difficulty and reputation of winning.")
    is_offline: bool = Field(description="True if physical attendance is required, False if fully online/virtual.")
    is_paid: bool = Field(description="True if there is an application or registration fee, False if free.")
    previous_year_winners_info: Optional[str] = Field(None, description="Brief note on historical winners if mentioned, otherwise null.")

# --- Extraction Function ---
def extract_opportunity(raw_web_text: str) -> Opportunity:
    prompt = f"Extract all relevant opportunity details from the following web page raw text:\n\n{raw_web_text}"

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert academic counselor. Extract opportunity details and format them strictly."
            },
            {"role": "user", "content": prompt}
        ],
        response_format=Opportunity
    )
    return completion.choices[0].message.parsed

# --- Run the code ---
if __name__ == "__main__":
    sample_text = """
    === GLOBAL TEEN INVENTORS SUMMIT 2026 ===
    Join the ultimate virtual pitch-off for future builders.
    URL: https://teeninventors.org/summit2026
    
    Are you a high school student with a tech product or a business idea?
    The Teen Inventors Summit is 100% online this year. Free to register.
    Open to students in grades 9 through 12. Must be residing in North America or Europe. 
    Ages 14-19 are welcome. English language.
    
    Must submit:
    1. A working software prototype (GitHub repository link is mandatory).
    2. A 3-minute video pitch.
    
    Our judges value technical code skills and presentation highly, but do not look at research papers.
    """

    print("Sending text to OpenAI...")
    result = extract_opportunity(sample_text)
    print("\n--- EXTRACTED DATA ---")
    print(result.model_dump_json(indent=2))