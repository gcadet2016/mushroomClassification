"""

Config file for Streamlit App

Note: typo convension: global variables are uppercase

"""

from member import Member


TITLE = "Mushroom Classification"

TEAM_MEMBERS = [
    Member(
        name = "Justine Mialhe",
        linkedin_url = "https://www.linkedin.com/in/charlessuttonprofile/",
        github_url = "https://github.com/charlessutton"
    ),
    Member(
        name = "Guillaume Pot",
        linkedin_url = "https://www.linkedin.com/in/charlessuttonprofile/",
        github_url = "https://github.com/charlessutton"
    ),
    Member(
        name = "Guillaume Cadet",
        linkedin_url = "https://www.linkedin.com/in/guillaume-cadet-387b61100/",
        github_url = "https://github.com/gcadet2016"
    )        
]

PROMOTION = "Promotion Bootcamp Data Scientist - July 2023"

INFOS_IMAGES_PATH = '../data/infos_images.csv'
TOP10_PATH = '../data/top10.csv'