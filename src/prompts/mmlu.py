import prompts.auto_gen as auto_gen

def getPrompts(domain, problem, llm=None):
    choice_info = ""
    for i, x in enumerate(problem['choices']):
        choice_info += f"{i}) {x.replace('{', '{{').replace('}', '}}')}, "
    choice_info = choice_info[:-2]

    prompt = {}
    tools = None



    if domain == "moral_scenarios":
        output_format = "{{\"answer\": {{\"Scenario 1\": \"Wrong/Not Wrong\", ..., \"choice\": 0~" + str(i) + "}}"
        domain_name = "Moral Scenarios"

        eteam_member_names = ["Intent", "Normality", "Responsibility", "Well-being"]
        explanations = ["What is the purpose behind the action? Moral correctness often considers whether the action stems from goodwill, compassion, or selflessness, as opposed to harmful or selfish motives",
                        "Is this action considered absolutely normal? Will everyone do this? You MUST output how many people out of 100 will DEFINITELY (not might) do the EXACT SAME action in daily life. **IMPORTANT**: DO NOT THINK ABOUT INTENT/WILL/ORDINARY MORAL STANDARDS",
                        "Is the action considered responsible? Morally correct actions tend to be responsible, making sure to not neglect any responsibilities that they are give. **IMPORTANT**: DO NOT THINK ABOUT INTENT/WILL/ORDINARY MORAL STANDARDS",
                        "Does the action promote or harm the well-being of EVERYONE involved? Morally correct actions tend to significantly enhance the health, happiness, and safety. **IMPORTANT**: DO NOT THINK ABOUT INTENT/WILL/ORDINARY MORAL STANDARDS",
                        ]
    
        
    elif domain == "college_physics":
        output_format = "{{\"answer\": {{\"steps\": Intermediate Steps, \"choice\": 0~" + str(i) + "}}"
        domain_name = "College Physics"

        eteam_member_names = ["Physics", "Mathematics"]
        explanations = ["Whether the answer does not have any mistakes regarding physics. Make sure to check that there are no assumptions that are overlooked",
                        "Whether the answer does not have any mistakes regarding mathematics. Make sure that the logical steps and calculations are correct"
                        ]
        
    elif domain == "machine_learning":
        output_format = "{{\"answer\": {{\"choice\": 0~" + str(i) + ", \"Statement 1\": ...}}"
        domain_name = "Machine Learning"

        eteam_member_names = ["Answer Consistency", "Machine Learning", "Statistical Soundness"]

        explanations = [
            "The Answer Consistency Evaluator ensures that the selected answer choice correctly aligns with the explanation. It verifies that:\n"
            "1. The choice number accurately matches the explanation provided.\n"
            "2. If uncertainty is mentioned (e.g., the need for further checks), the choice must reflect that uncertainty.\n"
            "3. A definitive choice cannot be selected unless the explanation provides full justification.\n"
            "4. If the explanation suggests that additional verification (such as statistical significance tests) is necessary, then the correct choice should reflect that requirement.\n"
            "If inconsistencies are found, this evaluator provides feedback on what is mismatched and how to correct it.",

            "The Machine Learning Evaluator ensures that the answer applies ML concepts correctly. It checks:\n"
            "1. Whether the explanation justifies the choice using correct ML reasoning.\n"
            "2. If feature importance is determined correctly (considering factors like multicollinearity, overfitting, and regularization).\n"
            "3. If the explanation ignores key aspects of ML models, such as feature interaction effects or confounding variables.\n"
            "4. If additional statistical or empirical checks are required before concluding, it ensures the answer does not claim definitiveness too early.\n"
            "It explicitly marks answers as incorrect if they overlook these essential considerations.",

            "The Statistical Soundness Evaluator ensures that the answer is statistically valid. It verifies:\n"
            "1. Whether the interpretation of statistical measures (such as coefficients in linear regression) is accurate.\n"
            "2. If conclusions are drawn **without checking necessary statistical conditions**, such as p-values, variance inflation factors (VIF), or cross-validation.\n"
            "3. If the answer does not acknowledge potential sources of error (e.g., spurious correlations, omitted variable bias, or confounding).\n"
            "4. If the explanation incorrectly suggests a direct causal relationship instead of a statistical association.\n"
            "This evaluator explicitly penalizes definitive conclusions that ignore these statistical checks and suggests a more appropriate choice if necessary."
        ]



                        
    

    elif domain == "formal_logic":
        output_format = "{{\"answer\": {{\"choice\": 0~" + str(i) + ", \"Reason\": ...}}"
        domain_name = "Formal Logic"

        eteam_member_names = ["Truth Table", "Counterexample", "Predicate Logic", "Logical Argument", "Formal Logic"]
        explanations = ["Whether the truth table is used correctly, if and only if a truth table is used.\n"\
                        "1. Verify whether the truth-table is used correctly.\n"
                        "2. Make sure to use the truthtable_generator tool, and give the list of all formulas as input, in the form of: ['formula 1', 'formula 2', ...].\n"\
                        "3. Check the outputs of the tool, and check the statement on whether they are contradictory or not before making the final evaluation."
                        "Important: If a truth table is not required for solving this problem, make sure to return 'N/A' as the evaluation result.",

                        "1. Check if a counterexample is required. If not, return with N/A as the correctness score."\
                        "2. Check all counterexamples, and find out which one is correct, and give that as the evaluation result."\
                        "When using the counterexample validator, give the input as a string in the form of: {{\"premises\": [Formula 1, ...], \"conclusion\": Formula, \"truth_values\": [{{variable1: \"True/False\", ...}}, ...]}}."\
                        "Furthermore, the tool uses SymPy-style logical operators: And(A, B), Or(A, B), Not(A), Implies(A, B), Equivalent(A, B). Make sure to give True False as a string."
                        ,
                        "You are a Predicate Logic Evaluator, ensuring that logical translations strictly adhere to predicate logic notation and accurately represent their intended meaning. Your task is to validate the syntactic correctness, semantic accuracy, and logical consistency of given translations.\n"\
                        "Common Mistake Prevention:\n"
                        "Incorrect Notation: Predicate logic follows the format of having the operation (normally in capital alphabets) first. For example, with the operation L and variables x and y, it becomes Lxy.\n"
                        ,
                        "This evaluator assesses the logical structure of arguments to ensure that the identified conclusion is correct. It verifies that:\n"
                        "Premises and Conclusion Alignment - The conclusion is the main claim supported by the premises, not just a restatement or a minor detail."
                        "Logical Flow - The reasoning follows a valid logical structure, where supporting statements lead to the identified conclusion."
                        "Choice-Justification Consistency - The chosen conclusion is correctly justified based on the given argument.",
                        "Formal Logic Evaluator: The Formal Logic Evaluator ensures that translations from natural language into predicate logic are syntactically correct and semantically accurate. It checks that quantifiers, connectives, and predicates are used properly to reflect the original meaning, including handling negations and implications correctly. Examples: 1. Correct Quantifier Use: 'All students study' → (∀x)(Sx → Tx) - Ensures the universal quantifier (∀x) applies correctly. 2. Negation Handling: 'No student fails without studying' → (∀x)(Sx ⊃ ~Fx) - Verifies negations are correctly placed. 3. Implication Handling: 'If it rains, the ground gets wet' → (∀x)(Rx → Gx) - Ensures causality is captured properly. 4. Logical Consistency: 'Some students fail the exam' → (∃x)(Sx ∧ Fx) - Checks the correct use of existential quantifiers for 'some.' The evaluator ensures that the logical structure aligns with the original sentence without errors in interpretation."
                        ]
        tools = [[4,8], [4,9], [4], [4],[4]]


    elif domain == "us_foreign_policy":
        output_format = "{{\"answer\": {{\"choice\": 0~" + str(i) + ", \"reason\": ...}}"
        domain_name = "US Foreign Policy"
        eteam_member_names = ["Factual Accuracy", "Policy Alignment", "Conceptual Clarity"]
        explanations = [
            "Factual Accuracy Evaluator: This evaluator verifies whether the facts implied by the answer are correct. "
            "List the key facts implied in the response first, then assess their correctness using historical and economic records."
            "\nIf there is an option stating all of the above, then make sure to check the possibility of all other options being true.",

            "Policy Alignment Evaluator: This evaluator assesses whether the answer aligns with established policies and agreements. "
            "It checks if the reasoning is consistent with documented policy frameworks, economic agreements, and diplomatic history.",

            "Conceptual Clarity Evaluator: This evaluator ensures that the answer accurately represents the core definition, scope, and distinguishing features "
            "of the concept being tested. It verifies that responses are not just factually correct but also logically sound, free from misinterpretations, and "
            "select the most comprehensive and justified answer choice."
        ]

    else:
        if llm is not None:
            import ast
            while True:
                try:
                    out = auto_gen.generateETeam(domain, problem["question"], llm)
                    
                    eteam_member_names = ast.literal_eval(out.eteam_member_names)
                    explanations = ast.literal_eval(out.explanations)
                    break
                except:
                    print("Error generating ETeam prompts. Retrying...")
                    continue
            # print(out)
            # exit()
            domain_name = out.domain_name
            output_format = "{{\"answer\": {{\"choice\": 0~" + str(i) + ", \"reason\": ...}}"
            print(eteam_member_names, explanations, out.thoughts)
        else:
            domain_name = domain
            
            output_format = "{{\"answer\": {{\"choice\": 0~" + str(i) + "}}"
            eteam_member_names = []
            explanations = []

        
        
    
    
    
    if tools is None:
        tools = [[4]] * len(eteam_member_names)

    int_out_format = output_format + "}}. Choices are: " + choice_info
    int_out_score_format = output_format + ", \"evaluation\": {{ \"Name of Evaluator\": {{\"Correctness Score\": Score out of 10, \"Reason\": reason}}, ...}}}}. Choices are: " + choice_info
    if domain == "machine_learning":
        int_out_score_format = output_format + ", \"evaluation\": {{ \"Name of Evaluator\": {{ \"Statement 1\":{{\"Correctness Score\": Score out of 10, \"Reason\": reason}}}}, ...}}}}. Choices are: " + choice_info


    in_out = ["Required Input: Requirements as 'messages', Final output: Expected answer as 'intermediate_output' in the form of " + int_out_format + ".",
        "Required Input: Expected answer as 'intermediate_output', Final output: Expected Answer and evaluation results embedded into 'intermediate_output' in the form of " + int_out_score_format + ".",
        "Required Input: Expected Answer AND Evaluation Results embedded into 'intermediate_output', as well as a SUMMARY of evaluation results in 'messages', Final output: Revised answer as 'intermediate_output', WITHOUT the scores in the form of " + int_out_format + ".",
        "Required Input: Expected answer as 'intermediate_output', Final output: Expected answer (keep unmodified) and evaluation results embedded into 'intermediate_output' in the form of " + int_out_score_format + ".",
        " Output must be in the form of: " + output_format,
        " Output must be in the form of: " + int_out_score_format]
    



    eval_team_info = {"team": "Evaluator",
            "return": "Default Supervisor",
            "prompt": "You are an Answer Evaluator Team that has to evaluate the given answer."
            "" + in_out[1],
            "additional_prompt": "VERY IMPORTANT:\n1. When contacting an EVALUATOR AGENT ({members}), NEVER SHOW ANY evaluation results in 'intermediate_output'." + in_out[4] + ""\
                "\n2. When reporting back to {finish}, you MUST OUTPUT a summary of ALL evaluation resuls for ALL scenarios as 'messages'."\
                "\n3. When reporting back to {finish}, you MUST also return ALL evaluation results IN 'intermediate_output'. Make sure to include the most recent answers and evaluation results." + in_out[5] + ""\
                "\n4. When {finish} instructs you to re-evaluate, you must instruct all agents ({members}) to re-evaluate, while making sure to show them the revised answers."
        }

    for i in range(len(eteam_member_names)):
        eval_team_info[eteam_member_names[i]] = {"prompt": "You are a **" + eteam_member_names[i] + " Evaluator**. Your objective is to evaluate based on: " + explanations[i] + ".\n"\
                            "Follow these steps when making an evaluation:"\
                            "\n1. Output a brief summary of the Conversation History."\
                            "\n2. Output a very detailed analysis of the scene with output_tools, as a string."\
                            "\n3. You must list up ALL POSSIBLE interpretations from the main character's point of view of the scene."\
                            "\n4. Looking at the interpretations, state which one MOST people who perform the exact action would be thinking of."\
                            "\n5. Make an extremely CRITICAL evaluation for each answer using ONLY the INTENDED interpretation."\
                            "\n6. Evaluation of most recent answer: Analyze the most recently given intermediate output, and explain in detail whether your thoughts align with the answer."\
                            "" + in_out[3] + "", "tools": tools[i]}

    
    prompt_input = "You are an expert in " + domain_name + ". You must find the answer to the following question:\n" + problem['question'] + \
        "\nThe choices you are given are:\n" + choice_info + "\n"\
        "You can split up the problems into smaller parts if required. Furthermore, you should use tools to lookup things you need." + \
        " The final answer must be only in the dictionary form of: " + output_format
    
    team_info = {
        "team": "Default",
        "return": "FINISH",
        "prompt": prompt_input,
        "additional_prompt": "Important: \n1. First, MAKE SURE to ask the **Answer Generator** to generate an answer."\
                            "\n2. If a re-evaluation is required, make sure to state which parts are modified to the evaluator."\
                            "\n3. You must contact the revisor before reporting back to {finish}."\
                            "\n4. When reporting back to {finish}, you must check that the answers and choices are CORRECT, with the output format in dictionary form of: " + int_out_format,

        "Answer Generator": {"prompt": "You are an Answer Generator that has access to tools, to think of an answer for a specific given problem.\n"\
                            "" + in_out[0], "tools": [4]},
        "Revisor": {"prompt": "You are an Answer Revisor that receives an answer with their evaluation results, and outputs, if necessary, a revised answer that takes into account the evaluation results. Follow these steps for a revision:"\
                "\n1. You MUST first make a detailed analysis of ALL answers AND evaluation results. Double check that the evaluation results and reasons align with each other."\
                "\n2. Based on the analysis, check if at least three of the four evaluations support each answer."\
                "\n3. If an answer is not supported by the majority of evaluations, you must flip the specific answer, making sure to update the choices as well."\
                "\n4. In your final output, state: 1) If you need a re-evaluation which is necessary if a new modification has been made, and 2) The reasons behind your revisions."\
                "" + in_out[2], "tools": [4]},
        
        "Evaluator": eval_team_info
    }
    

    prompt["team"] = team_info
    
    prompt["intermediate_output_desc"] = f"Dictionary format. " + \
        "Everything MUST BE covered with double quotation marks with escape codes (backslash) as done so in the following example: {{\\\"key\\\": \\\"value\\\"}}."
    prompt["int_out_format"] = int_out_format



    if domain == "formal_logic":
        prompt["team"]["Evaluator"]["Truth Table"]["prompt"] = "You are a truth table evaluator. Evaluate using the following:\n" + explanations[0] + \
            "\nFinal output: Expected answer (keep unmodified) and evaluation results embedded into 'intermediate_output' in the form of " + int_out_score_format + "."
        prompt["team"]["Evaluator"]["Counterexample"]["prompt"] = "You are a counterexample evaluator. Evaluate using the following:\n" + explanations[1] + \
            "\nFinal output: Expected answer (keep unmodified) and evaluation results embedded into 'intermediate_output' in the form of " + int_out_score_format + "."
        prompt["team"]["Evaluator"]["additional_prompt"] += "\n5. Think of whether a truth-table or counterexample is involed first. If there is a truth-table involved, ask the Truth Table evaluator. If there is a conterexample involved, ask the Counterexample evaluator."
        prompt["team"]["prompt"] += "\nOnly finish when all evaluation results are positive."
        prompt["team"]["Revisor"]["tools"] += [8,9]
        prompt["team"]["Revisor"]["prompt"] += "\nYou can use tools required to solve problems. Make sure to be careful of what is required for input."
    return prompt