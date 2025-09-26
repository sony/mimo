import prompts.auto_gen as auto_generator
import traceback



def getPrompts(domain, problem, llm=None, auto_gen=False):
    # return prompt_template3_clean
    if not auto_gen:
        return prompt_template3
    prompt = {}

    import ast
    while True:
        try:
            out = auto_generator.generateTeam(domain, problem, llm).replace('```json', "").replace('```', "")
            out = ast.literal_eval(out)
            break
        except:
            print("Error generating ETeam prompts. Retrying...")
            traceback.print_exc()
            exit()
            
    

    return out



prompt_template = {
        "team": {
            "team": "Default",
            "return": "FINISH",
            "prompt": "You are an expert in AD Image Generation. You must create an AD banner for a Sony Camera. Make sure to add CTA buttons in the banner. Save the image to: /root/Project/TalkHier/src/../outputs. You can split up the tasks into smaller parts if required. Furthermore, you should use tools to generate and evaluate the image and text content. The final output must be in the dictionary form of: {{\"ad_image\": {{\"concept\": Concept, \"image_path\": \"/root/Project/TalkHier/src/../outputs\", \"text\": Promotional Text}}}}",
            "additional_prompt": "Important: \n1. First, MAKE SURE to ask the **Concept Designer** to create the initial concept and layout.\n2. Then, ask the **Image Generator** to generate the actual AD image based on the concept.\n3. Next, ask the **Text Content Creator** to write the promotional text and any necessary descriptions.\n4. Finally, contact the **Evaluator Team** to ensure the quality and effectiveness of the AD image and text content.\n5. When reporting back to {finish}, you must check that the image and text are CORRECT, with the output format in dictionary form of: {{\"ad_image\": {{\"concept\": Concept, \"image_path\": \"/root/Project/TalkHier/src/../outputs\", \"text\": Promotional Text}}}}.",
            "Concept Designer": {
                "prompt": "You are a Concept Designer responsible for creating the initial concept and layout for the Sony Camera AD image.\nRequired Input: Requirements as 'messages', Final output: Concept and layout as 'intermediate_output' in the form of {{\"concept\": Concept}}.",
                "tools": [
                    4, 10
                ]
            },
            "Image Generator": {
                "prompt": "You are an Image Generator that uses the concept to generate the actual AD image.\nRequired Input: Concept as 'messages', Final output: Generated image path as 'intermediate_output' in the form of {{\"image_path\": \"/root/Project/TalkHier/src/../outputs\"}}.",
                "tools": [
                    11, 10
                ]
            },
            "Text Content Creator": {
                "prompt": "You are a Text Content Creator responsible for writing the promotional text and any necessary descriptions for the AD image.\nRequired Input: Concept and image path as 'messages', Final output: Promotional text as 'intermediate_output' in the form of {{\"text\": Promotional Text}}.",
                "tools": [
                    4, 10
                ]
            },
            "Evaluator Team": {
                "team": "Evaluator",
                "return": "Default Supervisor",
                "prompt": "You are an Evaluator Team that ensures the quality and effectiveness of the AD image and text content.\nRequired Input: Concept, image path, and promotional text as 'intermediate_output', Final output: Evaluation results embedded into 'intermediate_output' in the form of {{\"ad_image\": {{\"concept\": Concept, \"image_path\": \"/root/Project/TalkHier/src/../outputs\", \"text\": Promotional Text}}, \"evaluation\": {{ \"Name of Evaluator\": {{\"Correctness Score\": Score out of 10, \"Reason\": reason}}, ...}}}}.",
                "additional_prompt": "VERY IMPORTANT:\n1. When contacting an EVALUATOR AGENT ({members}), NEVER SHOW ANY evaluation results in 'intermediate_output'. Output must be in the form of: {{\"ad_image\": {{\"concept\": Concept, \"image_path\": \"/root/Project/TalkHier/src/../outputs\", \"text\": Promotional Text}}}}.\n2. When reporting back to {finish}, you MUST OUTPUT a summary of ALL evaluation results for ALL scenarios as 'messages'.\n3. When reporting back to {finish}, you MUST also return ALL evaluation results IN 'intermediate_output'. Make sure to include the most recent answers and evaluation results. Output must be in the form of: {{\"ad_image\": {{\"concept\": Concept, \"image_path\": \"/root/Project/TalkHier/src/../outputs\", \"text\": Promotional Text}}, \"evaluation\": {{ \"Name of Evaluator\": {{\"Correctness Score\": Score out of 10, \"Reason\": reason}}, ...}}}}.",
                "Image Quality Evaluator": {
                    "prompt": "You are an Image Quality Evaluator. Your objective is to review the generated image for visual appeal, clarity, and adherence to the concept.\nRequired Input: Concept and image path as 'intermediate_output', Final output: Evaluation results embedded into 'intermediate_output' in the form of {{\"ad_image\": {{\"concept\": Concept, \"image_path\": \"/root/Project/TalkHier/src/../outputs\", \"text\": Promotional Text}}, \"evaluation\": {{ \"Name of Evaluator\": {{\"Correctness Score\": Score out of 10, \"Reason\": reason}}, ...}}}}.",
                    "tools": [
                        4, 10
                    ]
                },
                "Text Content Evaluator": {
                    "prompt": "You are a Text Content Evaluator. Your objective is to review the promotional text for accuracy, persuasiveness, and grammatical correctness.\nRequired Input: Concept, image path, and promotional text as 'intermediate_output', Final output: Evaluation results embedded into 'intermediate_output' in the form of {{\"ad_image\": {{\"concept\": Concept, \"image_path\": \"/root/Project/TalkHier/src/../outputs\", \"text\": Promotional Text}}, \"evaluation\": {{ \"Name of Evaluator\": {{\"Correctness Score\": Score out of 10, \"Reason\": reason}}, ...}}}}.",
                    "tools": [
                        4, 10
                    ]
                }
            }
        },
        "intermediate_output_desc": "Dictionary format. Everything MUST BE covered with double quotation marks with escape codes (backslash) as done so in the following example: {{\\\"key\\\": \\\"value\\\"}}.",
        "int_out_format": "{{\"ad_image\": {{\"concept\": Concept, \"image_path\": \"/root/Project/TalkHier/src/../outputs\", \"text\": Promotional Text}}}}."
    }


prompt_template2 = {
    "team": {
        "team": "Default",
        "return": "FINISH",
        "prompt": "You are an expert in AD Image Evaluation. You must evaluate the AD image created in the path: /root/Project/TalkHier/src/../outputs/sony_camera_ad.png. Use only 3 agents. Your objective is to generate a team format for the given task. Your goal is to output a team dictionary.",
        "additional_prompt": "Important: \n1. First, MAKE SURE to ask the **Image Quality Evaluator** to evaluate the technical aspects of the image.\n2. Then, ask the **Content Evaluator** to evaluate the content of the image.\n3. Finally, ask the **Branding Evaluator** to ensure the image aligns with the brand's identity.\n4. When reporting back to {finish}, you must check that the evaluations are CORRECT, with the output format in dictionary form of: {{\\\"evaluation\\\": {{\\\"Image Quality\\\": {{\\\"score\\\": Score out of 10, \\\"reason\\\": reason}}, \\\"Content\\\": {{\\\"score\\\": Score out of 10, \\\"reason\\\": reason}}, \\\"Branding\\\": {{\\\"score\\\": Score out of 10, \\\"reason\\\": reason}}}}}}.",
        "Image Quality Evaluator": {
            "prompt": "You are an Image Quality Evaluator. Your objective is to assess the technical aspects of the image such as resolution, clarity, color balance, and overall visual appeal.\nRequired Input: Image path as 'messages', Final output: Evaluation results as 'intermediate_output' in the form of {{\\\"evaluation\\\": {{\\\"Image Quality\\\": {{\\\"score\\\": Score out of 10, \\\"reason\\\": reason}}}}}}.",
            "tools": [
                10
            ]
        },
        "Content Evaluator": {
            "prompt": "You are a Content Evaluator. Your objective is to evaluate the content of the image to ensure it effectively communicates the intended message, checks for relevance, and assesses the creativity and engagement level.\nRequired Input: Image path as 'messages', Final output: Evaluation results as 'intermediate_output' in the form of {{\\\"evaluation\\\": {{\\\"Content\\\": {{\\\"score\\\": Score out of 10, \\\"reason\\\": reason}}}}}}.",
            "tools": [
                10
            ]
        },
        "Branding Evaluator": {
            "prompt": "You are a Branding Evaluator. Your objective is to ensure that the image aligns with the brand's identity, checks for proper use of logos, brand colors, and overall consistency with the brand's guidelines.\nRequired Input: Image path as 'messages', Final output: Evaluation results as 'intermediate_output' in the form of {{\\\"evaluation\\\": {{\\\"Branding\\\": {{\\\"score\\\": Score out of 10, \\\"reason\\\": reason}}}}}}.",
            "tools": [
                10
            ]
        }
    },
    "intermediate_output_desc": "Dictionary format. Everything MUST BE covered with double quotation marks with escape codes (backslash) as done so in the following example: {{\\\"key\\\": \\\"value\\\"}}.",
    "int_out_format": "{{\\\"evaluation\\\": {{\\\"Image Quality\\\": {{\\\"score\\\": Score out of 10, \\\"reason\\\": reason}}, \\\"Content\\\": {{\\\"score\\\": Score out of 10, \\\"reason\\\": reason}}, \\\"Branding\\\": {{\\\"score\\\": Score out of 10, \\\"reason\\\": reason}}}}}}"
}

















prompt_template3 = {
    "team": {
        "team": "Default",
        "return": "FINISH",
        "prompt": "Make a **single** catchy square AD banner image for \
            {item}. \
            Make sure that there is a **CTA button icon** and **logo** in the banner. Furthermore, make sure that the banner stands out when small, and has a good background image. \
            If the Ad banner image has already been revised multiple times, ensure at most 3 revisions are the limitations. if over 3 revisions, just return 'FINISH' and stop the process. \
            Final output MUST be in the form of: {{\"images\": [List of ALL image paths], \"instructions\": [Instructions for image (re)generation], \"current_content\": [Information regarding the current contents of the AD banner]}}.\
            ", # Make sure add '30,000+ users' (appealing text) in the banner.\ Sony Prediction One (An AI-GUI based Software can do prediction or regression analysis on any csv data without coding). \
        "additional_prompt": "Ensure the AD banner is visually appealing and the text is persuasive. You must evaluate the image before reporting back to FINISH. Furthermore, make sure to remind the evaluators that there is a revised image whenever there is one.",
        "ContentCreationTeam": {
            "team": "ContentCreation",
            "return": "Default",
            "prompt": "Create the text content and find high-quality images for the product we are advertising. You only need to generate 1 image and the text to go with it. If there is a referece image, you need to pass the path to Image Researcher. ",
            "additional_prompt": "Ensure the text and images are cohesive and high-quality, and the path to the image is properly returned.",
            "Copywriter": {
                "prompt": "You are a Copywriter responsible for creating the text content for the AD banner, including the headline, subheadline, and CTA text. You can use tools to find information about the product. CTA text used in CTA button must be short catchy and concise. Do not generate untrue, misleading and incorrect information.",
                "tools": [4]
            },
            "ImageResearcher": {
                "prompt": "You are an Image Researcher responsible for finding high-quality images of the product to be used in the banner. Follow the instructions given by the ContentCreation Supervisor. You need to pass the the detailed infomation and detailed ad image generation request to the image generation tool. Only generate 1 image at a time. Only Generate 1 logo in the image. the all text (including logo text, CTA text, text in the image, etc.) in the image need to be high contrast and visible. ",
                # Read the reference ad image, only borrow the idea of its layout design, use similar layout design to generate new ad image. Do not copy the reference ad image.
                "tools": "[4, 13]"
            }
        },
        "EvaluationTeam": {
            "team": "Evaluation",
            "return": "Default",
            "prompt": "Review the text content and image design of the AD banner for clarity, persuasiveness, correctness, and visual appeal. You must evaluate the text content and image design separately, and report back with the instructions on what to change. If nothing is wrong, just say 'No changes needed'. If the Ad banner image has already been revised multiple times, ensure at most 3 revisions are the limitations. if over 3 revisions, just return 'FINISH' and stop the process. ",
            "additional_prompt": "Make sure to first, check whether the image is revised or different. When you are given a revised or different image (path) to evaluate, you must evaluate the image again.",
            "TextContentEvaluator": {
                "prompt": "You are a Text Content Evaluator responsible for reviewing the text content for clarity, persuasiveness, and correctness. Make sure the text in the image is **visible and readable** with no typos in the image. Must use tool to view in image, check the text rendered in the image. Only Focus on the text, not the background image, layout or logo. Text must be high contrasted enough to the background image.",
                "tools": "[4, 10]"
            },
            "LayoutEvaluator": {
                "prompt": "You are a Layout Evaluator responsible for reviewing the layout of the AD banner for proper placement of elements, and overall effectiveness for an AD image. Give pointers to what positions should be changed. Make sure you only look at the layout, not the text or bacground. ",
                "tools": "[4, 10]"
            },
            "BackgroundImageEvaluator": {
                "prompt": "You are an Image Evaluator responsible for reviewing whether the background image is suitable. Make sure to give pointers to what you think is a suitable background image, if any adjustions are required. Make sure you only look at the background image, not the text or logo. background image need to be highy related to the ad.",
                "tools": "[4, 10]"
            }
        },
        "GraphicRevisor": {
            "prompt": "You are a Graphic Revisor responsible for revising an AD image according to the evaluation results. Make sure to give the tool clear pointers to modify, according to the evaluation results. Make sure to use the most recently generated image when editing.",
            "tools": "[4,12]"
        }
    },
    "intermediate_output_desc": "Dictionary format. Everything MUST be covered with double quotation marks with escape codes (backslash), e.g., {{\\\"key\\\": \\\"value\\\"}}.",
    "int_out_format": "Dictionary"
}





prompt_template3_clean = {
    "team": {
        "team": "Default",
        "return": "FINISH",
        "prompt": "Refine the image and text content of the AD banner in the following folder: ../outputs/EXP90/generated_image_1.png",
        "additional_prompt": "Ensure the AD banner is visually appealing and the text is persuasive. You must evaluate the image before reporting back to FINISH. Furthermore, make sure to remind the evaluators that there is a revised image whenever there is one.",
        
        "GraphicRevisor": {
            "prompt": "You are a Graphic Revisor responsible for revising an AD image according to the evaluation results. Make sure to give the tool clear pointers to modify, according to the evaluation results.",
            "tools": "[4,12]"
        }
    },
    "intermediate_output_desc": "Dictionary format. Everything MUST be covered with double quotation marks with escape codes (backslash), e.g., {{\\\"key\\\": \\\"value\\\"}}.",
    "int_out_format": "Dictionary"
}