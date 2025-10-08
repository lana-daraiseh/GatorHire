import ollama
import re
from ollama import chat
from ollama import ChatResponse
import json

def extract_resume_with_ollama(resume_text: str, model: str = "gemma3") -> dict:
    """
    Extract resume info with ollama.
    """
        
    cont = f"""
    You are a resume parser. Extract structured information from the text below.
    Return JSON strictly in this schema:

    {{
      "contact": {{"name":"", "email":"", "phone":"", "links":[]}},
      "summary": "",
      "experience": [
        {{"title":"", "company":"", "dates":"", "bullets":[""]}}
      ],
      "projects": [{{"name":"", "description":""}}]
      "education": [{{"school":"", "degree":"", "GPA":"", "year":""}}],
      "skills": []
    }}

    Resume Text:
    {resume_text}
    """
    prompt = [{'role': 'user', 'content': cont}]

    response = ollama.chat(model = 'gemma3', messages = prompt)
    
    raw_output = response.message.content.strip()

    raw_output = re.sub(r"^```(?:json)?|```$", "", raw_output, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        print("Model did not return valid JSON, raw output:\n", raw_output)
        parsed = {"raw": raw_output}
    return parsed

# Test
if __name__ == "__main__":
    from tempParser import parse_resume 

    # resume_path = "Christian Schneider Resume.pdf"
    # resume_path = "Elliot Blain Computer Science Resume.pdf"
    #resume_path = "Anthony Franzino Resume 2025.pdf"
    resume_path = "Lana Daraiseh TEST Resume.pdf"
    parsed = parse_resume(resume_path)

    ollama_structured = extract_resume_with_ollama(parsed["text"])
    print(json.dumps(ollama_structured, indent=2, ensure_ascii=False))