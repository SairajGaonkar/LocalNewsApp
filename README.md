# LocalNewsApp

# Requirements
- Python3
- Flask
- NodeJS
- NPM

# References
- APIs Used:
  - Google news API: https://rapidapi.com/newscatcher-api-newscatcher-api-default/api/google-news
  - Twitter API: https://developer.twitter.com/en/docs/twitter-api/v1/tweets/search/api-reference/get-search-tweets
- Zipcodes library used in FetchZipcodeDetails.py: https://pypi.org/project/zipcodes/
- Database: socolab.luddy.indiana.edu:27017
 

# Setup
- Clone the repository.
```bash
git clone https://github.iu.edu/SoCo/LocalNewsApp.git
```
- Go to the project directory.
```bash
cd LocalNews
```

## Backend
- Use NewsAPI-BackEnd branch to get the backend code.
```bash
git checkout NewsAPI-BackEnd
```
- Create virtual environment.
```bash
python3 -m venv myenv
```
- Activate the virtual environment.

For macOS/Linux: 
```bash
source myenv/bin/activate
```
For windows:
```bash
.\env\Scripts\activate
```
- Install requirements.
```bash
pip install -r requirements.txt
```
- Run the project
```bash
python3 app.py
(or)
SET FLASK_APP=app.py
flask run
```

## Frontend
- Use UI branch to get the frontend code.
```bash
git checkout UI
```
- Install expo-CLI (In your project directory).
```bash
npm install expo-cli
```
- Install necessay node-modules.
```bash
expo install
```
- Run the project.
```bash
expo start
```

