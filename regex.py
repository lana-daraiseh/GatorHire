import re

class Regex:

    EMAIL = re.compile(
        r"[A-Za-z0-9._\\-]+@[A-Za-z0-9-]*\\.[a-z]{2,3}"
    )

# check this one
    PHONE = re.compile(
        r"(\+?\d{1,3})?[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}"
    )

    NAME = re.compile(
        r"^[A-Za-z]+(?:\s[A-Za-z]+)+", re.MULTILINE
    )

    SKILLS = [
        "Python", "Java", "C++", "SQL", "Machine Learning", "Data Analysis",
        "Leadership", "React", "MATLAB", "Git", "C#"
    ]

    SECTION_HEADINGS = [
        "Experience", "Work Experience", "Employment History", "Education",
        "Skills", "Projects", "Certifications", "Technical Skills"
    ]

    YEAR = re.compile(
        #r"\d{4}", re.IGNORECASE
        r"(19|20)\d{2}"

    )

    URL = re.compile(
            "https?://[A-Za-z]+\\.[a-z]{2,3}"
    )

    WHITESPACE = re.compile(
         "[ \b\n\r\t]+"
    )
