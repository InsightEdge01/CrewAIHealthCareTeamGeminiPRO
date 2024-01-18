from crewai import Agent, Task, Crew, Process
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
import gradio as gr

# Set gemini pro as llm
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose=True,
                             temperature=0.5,
                             google_api_key="")

duckduckgo_search = DuckDuckGoSearchRun()

def create_crewai_setup(age, gender, disease):
    # Define Agents
    fitness_expert = Agent(
        role="Fitness Expert",
        goal=f"""Analyze the fitness requirements for a {age}-year-old {gender} with {disease} and 
                 suggest exercise routines and fitness strategies""",
        backstory=f"""Expert at understanding fitness needs, age-specific requirements, 
                      and gender-specific considerations. Skilled in developing 
                      customized exercise routines and fitness strategies.""",
        verbose=True,
        llm=llm,
        allow_delegation=True,
        tools=[duckduckgo_search],
    )
    
    nutritionist = Agent(
        role="Nutritionist",
        goal=f"""Assess nutritional requirements for a {age}-year-old {gender} with {disease} and 
                 provide dietary recommendations""",
        backstory=f"""Knowledgeable in nutrition for different age groups and genders, 
                      especially for individuals of {age} years old. Provides tailored 
                      dietary advice based on specific nutritional needs.""",
        verbose=True,
        llm=llm,
        allow_delegation=True,
    )
    
    doctor = Agent(
        role="Doctor",
        goal=f"""Evaluate the overall health considerations for a {age}-year-old {gender} with {disease} and 
                 provide recommendations for a healthy lifestyle.Pass it on to the
                  disease_expert if you are not an expert of {disease} """,
        backstory=f"""Medical professional experienced in assessing overall health and 
                      well-being. Offers recommendations for a healthy lifestyle 
                      considering age, gender, and disease factors.""",
        verbose=True,
        llm=llm,
        allow_delegation=True,
    )

    # Check if the person has a disease
    if disease.lower() == "yes":
        disease_expert = Agent(
            role="Disease Expert",
            goal=f"""Provide recommendations for managing {disease}""",
            backstory=f"""Specialized in dealing with individuals having {disease}. 
                          Offers tailored advice for managing the specific health condition.
                          Do not prescribe medicines but only give advice.""",
            verbose=True,
            llm=llm,
            allow_delegation=True,
        )
        disease_task = Task(
            description=f"""Provide recommendations for managing {disease}""",
            agent=disease_expert,
            llm=llm
        )
        health_crew = Crew(
            agents=[fitness_expert, nutritionist, doctor, disease_expert],
            tasks=[task1, task2, task3, disease_task],
            verbose=2,
            process=Process.sequential,
        )
    else:
        # Define Tasks without Disease Expert
        task1 = Task(
            description=f"""Analyze the fitness requirements for a {age}-year-old {gender}. 
                            Provide recommendations for exercise routines and fitness strategies.""",
            agent=fitness_expert,
            llm=llm
        )

        task2 = Task(
            description=f"""Assess nutritional requirements for a {age}-year-old {gender}. 
                        Provide dietary recommendations based on specific nutritional needs.
                        Do not prescribe a medicine""",
            agent=nutritionist,
            llm=llm
        )

        task3 = Task(
            description=f"""Evaluate overall health considerations for a {age}-year-old {gender}. 
                        Provide recommendations for a healthy lifestyle.""",
            agent=doctor,
            llm=llm
        )
        
        health_crew = Crew(
            agents=[fitness_expert, nutritionist, doctor],
            tasks=[task1, task2, task3],
            verbose=2,
            process=Process.sequential,
        )

    # Create and Run the Crew
    crew_result = health_crew.kickoff()

    # Write "No disease" if the user does not have a disease
    if disease.lower() != "yes":
        crew_result += f"\n disease: {disease}"

    return crew_result

# Gradio interface
def run_crewai_app(age, gender, disease):
    crew_result = create_crewai_setup(age, gender, disease)
    return crew_result

iface = gr.Interface(
    fn=run_crewai_app, 
    inputs=["text", "text", "text"], 
    outputs="text",
    title="CrewAI Health,Nutriion and Fitness Analysis",
    description="Enter age, gender, and disease (or 'no' if there is no disease) to analyze fitness, nutrition, and health strategies."
)

iface.launch()
