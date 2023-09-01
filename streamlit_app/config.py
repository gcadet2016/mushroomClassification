"""

Config file for Streamlit App

"""

from member import Member


TITLE = "My Awesome App"

TEAM_MEMBERS = [
    Member(
        name="John Doe",
        linkedin_url="https://www.linkedin.com/in/guillaume-cadet-387b61100/",
        github_url="https://github.com/gcadet2016",
    ),
    Member("Jane Doe"),
]

PROMOTION = "Promotion Bootcamp Data Scientist - July 2023"
