from agno.agent import Agent

from agno.models.google import Gemini

model = Gemini(api_key="AIzaSyAmW-JO8ikezOZ16mQBHnCVJZ4Tvxx7eaU")

# Create an agent with the model and instructions
agent = Agent(

model=model,

instructions=["Rephrase the following sentence in a more formal tone."],

markdown=True

)


response = agent.run("Hey, just letting you know I won’t make it to the meeting later — something came up.")

# Print the response
print(response.content)