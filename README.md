# Academic World Keyword Data Dashboard

This project is about building a dashboard that allows the user to explore different keywords in the academic world database. The target users for this project are students or researchers who are interested in finding relevant information about keywords they are interested in. And the objective is to provide relevant information such as top publications, professors, and institutes for given keywords.

## Database Connection

- MySQL
  - Follow MP3's instruction to import data.
  - Constraints:
    - Table that uses to record university/institute information is named "Institute" and has a column for "name".
    - Table that uses to record keyword information is named "keyword" and has a column for "name".
  - A new table will be created to store user's favorite keywords
- MongoDB
  - Follow MP3's instruction to import data.
- Neo4j
  - Follow MP3's instruction to import data.
- Fill in necessary information in `secret.py`

## Setup

- Start MySQL server and Neo4j server
- Make sure to have pipenv installed
- Run `pipenv shell` to activate the shell
- Run `python app.py` to start the app
