from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate

class routeResponse(BaseModel):
    # output_format: str = Field(description="The intermediate output format for each evaluator agent. Must be built on top of: " + "{{\"answer\": {{\"choice\": ..., \"reason\": ...}}")
    thoughts: str = Field(description="Thoughts on what type of evaluators you would need, and why.")
    domain_name: str = Field(description="Format the domain name to be proper.")
    eteam_member_names: str = Field(description="The list of evaluators to deploy. Format: [\"Evaluator1\", \"Evaluator2\", ...], with specific names for each agent. Make sure to output as a list.")
    explanations: str = Field(description="The list of the explanations of evaluators to deploy. Format: [\"Explanation1\", \"Explanation2\", ...]. Make sure to output as a list.")

def generateETeam(domain_name, problem, llm):
    prompt_list = [("system", 
                    "You are an Evaluation Team generator for the task:" + domain_name + ". The question is: " + problem + "."
                    " Your objective is to generate novel and specific evaluators that evaluate given answers generated for the above problem, and provide explanations of their role."
                    " Make sure to give each evaluator a specific name representing the type of evaluation they are doing. Your outputs are:\n"
                    "thoughts: "
                    "Output your detailed thoughts on what kind of evaluators are needed, and why they are needed.\n"
                    "eteam_member_names: "
                    "Generate multiple evaluators to deploy for this task. Format should be [\"Evaluator1\", \"Evaluator2\", ...], with specific names for each agent. Make sure to output as a list, as a string, without using special characters.\n"
                    "explanations: "
                    "Generate the explanations for the above evaluators.. Format should be [\"Explanation1\", \"Explanation2\", ...]. Make sure to output as a list, as a string.\n"
                    "domain_name: "
                    "Format the domain name to be proper.\n"
                    "Make sure to name each evaluator, and not give Evaluator 1 as output, and provide up to 3 evaluators. "
                    "Each evaluator must have a very specific role, and must give completely different insights into the problem given. "
                    "The evaluators must be able to confidently spot any mistake in the given answer, without taking in bias from the given results. "
                    "Furthermore, try to make the explanations as specific as possible, so that any mistakes in each area can be pointed out."
                    )]
    prompt =  ChatPromptTemplate.from_messages(prompt_list)
    chain = prompt | llm.with_structured_output(routeResponse, method='function_calling')
    return chain.invoke(input={}, state={})




class routeResponse2(BaseModel):
    # output_format: str = Field(description="The intermediate output format for each evaluator agent. Must be built on top of: " + "{{\"answer\": {{\"choice\": ..., \"reason\": ...}}")
    domain_name: str = Field(description="Format the domain name to be proper.")
    team: str = Field(description="Output dictionary of a team.")



example_dict = {
    "team": {
        "team": "Default",
        "return": "FINISH",
        "prompt": "Make a **single** square AD banner image for \
            {item}. \
            Make sure that there is a **CTA button icon** and **logo** in the banner. \
            Final output must be in the form of: {{\"generated_regenerated_images\": [List of ALL image paths], \"instructions\": [Instructions for image (re)generation], \"current_content\": [Information regarding the current contents of the AD banner]}}.\
            ", # Make sure add '30,000+ users' (appealing text) in the banner.\ Sony Prediction One (An AI-GUI based Software can do prediction or regression analysis on any csv data without coding). \
        "additional_prompt": "Ensure the AD banner is visually appealing and the text is persuasive. You must evaluate the image before reporting back to FINISH. Furthermore, make sure to remind the evaluators that there is a revised image whenever there is one.",
        "ContentCreationTeam": {
            "team": "ContentCreation",
            "return": "Default",
            "prompt": "Create the text content and find high-quality images for the product we are advertising. You only need to generate 1 image and the text to go with it.",
            "additional_prompt": "Ensure the text and images are cohesive and high-quality.",
            "Copywriter": {
                "prompt": "You are a Copywriter responsible for creating the text content for the AD banner, including the headline, subheadline, and CTA text",#. You can use tools to find information about the product.",
                "tools": [4]
            },
            "ImageResearcher": {
                "prompt": "You are an Image Researcher responsible for finding high-quality images of the product to be used in the banner. You need to pass the the detailed infomation and detailed ad image generation request to the image generation tool. Only generate 1 image at a time.",
                "tools": "[13]"
            }
        },
        "EvaluationTeam": {
            "team": "Evaluation",
            "return": "Default",
            "prompt": "Review the text content and image design of the AD banner for clarity, persuasiveness, correctness, and visual appeal. You must evaluate the text content and image design separately.",
            "additional_prompt": "DO NOT implement any changes to the images, and only report the evaluation results back to the default supervisor. You may be given a different image to evaluate each turn, so be aware.",
            "TextContentEvaluator": {
                "prompt": "You are a Content Evaluator responsible for reviewing the text content for clarity, persuasiveness, and correctness. Make sure the text generated by the Copywriter is **visible and readable** with no typos in the image. Make sure you only look at the text, not the background image or logo.",
                "tools": "[4, 10]"
            },
            "LayoutEvaluator": {
                "prompt": "You are a Layout Evaluator responsible for reviewing the layout of the AD banner for proper placement of elements, and overall effectiveness for an AD image. Give pointers to what positions should be changed. Make sure you only look at the layout, not the text or logo.",
                "tools": "[4, 10]"
            },
            "BackgroundImageEvaluator": {
                "prompt": "You are an Image Evaluator responsible for reviewing whether the background image is suitable. Make sure to give pointers to what you think is a suitable background image, if any adjustions are required. Make sure you only look at the background image, not the text or logo.",
                "tools": "[4, 10]"
            }
        },
        "GraphicRevisor": {
            "prompt": "You are a Graphic Revisor responsible for revising an AD image according to the evaluation results. Make sure to give the tool clear pointers to modify, according to the evaluation results.",
            "tools": "[4,12]"
        }
    },
    "intermediate_output_desc": "Dictionary format. Everything MUST be covered with double quotation marks with escape codes (backslash), e.g., {{\\\"key\\\": \\\"value\\\"}}.",
    "int_out_format": "Dictionary"
}

example_dict2 = {
    "team": {
        "team": "{team_name}",
        "return": "FINISH",
        "prompt": "{main_task_prompt}",
        "additional_prompt": "{optional_additional_prompt}",

        "{agent_1_name}": {
            "prompt": "{agent_1_prompt_description}",
            "tools": "[{tool_ids_for_agent_1}]"
        },

        "{agent_2_name}": {
            "prompt": "{agent_2_prompt_description}",
            "tools": "[{tool_ids_for_agent_2}]"
        },

        # Add more agents here as needed
        "{agent_N_name}": {
            "prompt": "{agent_N_prompt_description}",
            "tools": "[{tool_ids_for_agent_N}]"
        },

        "{subteam_name}": {
            "team": "{subteam_type}",
            "return": "{upper_team_supervisor}",
            "prompt": "{subteam_prompt}",
            "additional_prompt": "{subteam_additional_prompt}",

            "{subagent_1_name}": {
                "prompt": "{subagent_1_prompt}",
                "tools": "[{tool_ids_for_subagent_1}]"
            },

            "{subagent_2_name}": {
                "prompt": "{subagent_2_prompt}",
                "tools": "[{tool_ids_for_subagent_2}]"
            }

            # Add more subagents as needed
        }
    },

    "intermediate_output_desc": (
        "Dictionary format. Everything MUST be covered with double quotation marks "
        "with escape codes (backslash), e.g., {{\\\"key\\\": \\\"value\\\"}}."
    ),

    "int_out_format": (
        "{output_format}"
    )
}



tools = """ID	Name	Explanation
0	getSerpTool()	Interfaces with a SERP API to fetch live search results. Useful for real-time web data.
1	CustomCounterTool()	Counts items or keywords in input. Useful for frequency analysis or metric reporting.
4	OutputTool()	Formats or processes final output. Likely wraps results for presentation or export.
5	ClickAggregator	Aggregates and analyzes click data from a keyword dataset. Helps assess keyword performance.
6	python_repl	A Python REPL shell that executes Python commands dynamically. Useful for computation.
7	RejectWordTool	Filters or rejects inputs containing blacklisted words from a CSV. Used for moderation.
8	TruthTableGenerator	Generates truth tables for logical expressions. Helpful in logic-related tasks.
9	CounterexampleVerifier	Verifies logic or rule correctness by testing counterexamples. Checks for logical flaws.
"""

tools = """ID	Name	Explanation
0	getSerpTool()	Interfaces with a SERP API to fetch live search results. Useful for real-time web data.
4	OutputTool()	Formats or processes final output. Likely wraps results for presentation or export.
6	python_repl	A Python REPL shell that executes Python commands dynamically. Useful for computation.
10	ImageToBase64Tool	Converts images to base64 format. Useful for embedding images in text. Make sure to use this when evaluations on images are required.
12  ImageEditTool   Edits images using a image editing tool.
13  ImageGenTool   Generates images using a image generation tool.
14  FileReadWriterTool   Reads and writes content into a file.
"""

import json
def generateTeam(domain_name, problem, llm):
    prompt_list = [("system", 
                    "You are a Team generator for the task:" + domain_name + ". The question is: " + problem + ".\n"
                    "1. Think of what agents are required to solve the problem.\n"
                    "2. If multiple agents can be grouped together, make them a team. You can have multilpe layers of subteams if required.\n"
                    "3. Show connections between all agents, in a graph format.\n"
                    "4. All agents and teams are part of the Default team.\n"
                    "5. Try to add evaluators where possible.\n"
                    "Important: Make sure to keep the number of agents as minimal as possible, while trying to guarantee that the solution can be gained.\n"
                    "An Example of a team is: Default Team = [Generator, Revisor, Evaluator Team], Evaluator Team = [Evaluator 1, Evaluator 2]"
                    )]


    prompt = ChatPromptTemplate.from_messages(prompt_list)
    chain = prompt | llm
    out = chain.invoke(input={}, state={})
    print(out.content)

    prompt_list = [("system", 
                    "You are a Team generator for the task:" + domain_name + ". The question is: " + problem + "."
                    " Your objective is to generate a team format for the given task."
                    " Your goal is to output a team dictionary. The template of the format is as follows:\n"
                    "{example_dict}\n"
                    " The example is: \n"
                    "{example_dict2}\n"
                    ),
                    ("system",
                    "List of tools available:\n" + tools),
                    ("system",
                    "Make a team dictionary like the above, with the team members and names as shown in the following:\n"
                    + "{out_content}\n"
                    "Make sure to use double brackets for the dictionary, and escape all double quotes with backslashes. Furthermore, make sure to retain all elements in the dictionary, and not remove any elements."
                    " Finally, make sure the team names do not have spaces, and do not end with Team. Make sure every explanation is clear and consise so no misunderstandings will be made.")]
    # print(prompt_list)
    prompt =  ChatPromptTemplate.from_messages(prompt_list)
    # print(prompt.invoke(input={"example_dict":  json.dumps(example_dict, indent=4)}, state={"example_dict":  example_dict}))
    chain = prompt | llm#.with_structured_output(routeResponse2, method='function_calling')
    out = chain.invoke(input={"example_dict":  json.dumps(example_dict2, indent=4), "example_dict2":  json.dumps(example_dict, indent=4), "out_content": out.content}, state={})
    print(out.content)
    return out.content