CODE_INST = """You are an expert in qualitative research and grounded theory, and you are good at annotating the reasoning behaviors of language models' generated reasoning using a taxonomy of reasoning behaviors. 

You will be given two models' reasoning traces toward a question. 

You will also be given a reasoning taxonomy that illustrates the known reasoning traits and styles of different language models. 

Your task is to annotate the reasoning behaviors (in the taxonomy) that appeared in the given reasoning traces based on their definitions in the given taxonomy.

Think step by step. You should start from annotating both OUTPUTs using the reasoning behaviors in the taxonomy following the provided definitions.

You don't need to use every reasoning behavior in the reasoning taxonomy in your annotation. It's possible that some reasoning behaviors don't occur in any of the given outputs.

For example, given a reasoning taxonomy with N reasoning behaviors, your step-by-step chain of thoughts should look like this:

------
{Start by annotating the reasoning traces}
[Annotate the reasoning OUTPUT A with the given taxonomy]: [Beginning of the OUTPUT A] [Behavior Name for first paragraph / sentence] [Behavior Name for second paragraph / sentence] [continue for the rest paragraphs / sentences] ... [Behavior Name for last paragraph / sentence] [End of the OUTPUT A] {YOU MUST ANNOTATE THE WHOLE REASONING OUTPUT A}

[Annotate the reasoning OUTPUT B with the given taxonomy]: [Beginning of the OUTPUT B] [Behavior Name for first paragraph / sentence] [Behavior Name for second paragraph / sentence] [continue for the rest paragraphs / sentences] ... [Behavior Name for last paragraph / sentence] [End of the OUTPUT B] {YOU MUST ANNOTATE THE WHOLE REASONING OUTPUT B}

{You should annotate the OUTPUT sentence by sentence (or paragraph by paragraph if it's long). For each sentence / paragraph, represent it with one of the reasoning behaviors if applicable. Use [Not in Taxonomy] for behaviors not described by the given taxonomy. Don't be lazy even if the input outputs are long!}

{If the original OUTPUT A and OUTPUT B do not have the same length, your annotation for each output does not need to be in the same length as well.}
------

Make sure your output chain of thoughts follows this format exactly.

Below is the reasoning taxonomy that you will use for the classification,
"""

CORRECTION_INST = """You are an expert in qualitative research and grounded theory, and you are good at distinguishing the reasoning behaviors of different language models.

There is a reasoning taxonomy that outlines the distinguishing reasoning behaviors of various large language models. Previously, one could classify the author model of a reasoning output based on this reasoning taxonomy. However, this reasoning taxonomy cannot distinguish the new reasoning outputs provided by the user. 

Your task is to identify missing distinguishing reasoning behaviors and add them to the reasoning taxonomy so that one can accurately classify these new reasoning outputs. Focus on discovering diverse and unique reasoning traits that are not currently captured in the reasoning taxonomy.

You should think step by step when comparing two models' reasoning outputs. It is okay if an existing reasoning behavior does not appear in the provided output. 

If there are distinguishing differences in reasoning behaviors or language styles, but they are not included in the reasoning taxonomy, you should add a new reasoning behavior for each of those differences in the reasoning taxonomy. 

When adding the new reasoning behavior, you should provide a short name of the reasoning behavior with its detailed definition, such as [Reasoning behavior name]: [What this reasoning trait is about] [Example of this behavior quoted from the given outputs]. If the reasoning behavior name contains multiple words, add space between the words.

[Example of this behavior] can be a direct quote. Make sure it will give a different expert enough information to make the same decision as yours.

Follow the guidelines below when updating the reasoning taxonomy:

Give meaningful and detailed reasoning behavior definitions (at least 100 words) and examples. You want to make sure a different expert can make the same classification as you did using your reasoning taxonomy.

If you use words like "meticulous" or "detailed" in your reasoning behavior name or definition, explain in the reasoning behavior definition what they mean and what keywords signal this meticulousness. 

If you use words like "structured" or "systematic", then explain in the code definition what type of structure or framework the reasoning trace has. Be very detailed!

Make sure you compare the reasoning outputs with the existing reasoning behaviors first. Add new reasoning behaviors only if they are significantly different from the existing reasoning behaviors and their definitions. Do not add reasoning behaviors whose names and definitions are synonyms to the existing ones. It's okay if their examples of reasoning behavior are different.

Examples of reasoning behaviors include verification (error-checking), backtracking (abandoning failing approaches), backward chaining (reasoning from desired outcomes to initial inputs), and sub-goal setting (decomposing problems into smaller steps). 

Reasoning steps that you should analyze include problem definition, initial response, planning, execution and monitoring, reconstruction, and solution verification.

You should use them as guidelines but also do not limit your coding to these known categories.

For example, your step-by-step chain-of-thoughts should look like this:

------
[Start by annotating the reasoning traces]
[Annotate the reasoning OUTPUT A with the given taxonomy]: [Beginning of the OUTPUT A] [Behavior Name for first paragraph / sentence] [Behavior Name for second paragraph / sentence] [continue for the rest paragraphs / sentences] ... [Behavior Name for last paragraph / sentence] [End of the OUTPUT A] {YOU MUST ANNOTATE THE WHOLE REASONING OUTPUT A}

[Annotate the reasoning OUTPUT B with the given taxonomy]: [Beginning of the OUTPUT B] [Behavior Name for first paragraph / sentence] [Behavior Name for second paragraph / sentence] [continue for the rest paragraphs / sentences] ... [Behavior Name for last paragraph / sentence] [End of the OUTPUT B] {YOU MUST ANNOTATE THE WHOLE REASONING OUTPUT B}

{You should annotate the OUTPUT sentence by sentence (or paragraph by paragraph if it's long). For each sentence / paragraph, represent it with one of the reasoning behavior if applicable. If a behavior is not described by the taxonomy, give it a meaningful name and annotate it. However, this behavior may not be the missing behavior in our taxonomy, unless you find it occurs more often in one versus the other reasoning output. Don't be lazy even if the input outputs are long!}

{If the original OUTPUT A and OUTPUT B do not have the same length, your annotation for each output does not need to be the same length as well.}

Now, I will summarize my annotation for each OUTPUT, and then count number of behaviors occurred in each OUTPUT.

### [Existing reasoning behavior's name]
### [Definition of this reasoning behavior (reasoning behavior); which model exhibits this reasoning behavior; reasoning with the given output; which output shows this reasoning behavior (with one quote)]
### [Whether this reasoning behavior occurs in OUTPUT A: Either "This reasoning behavior is observed in OUTPUT A." or "This reasoning behavior is not observed in OUTPUT A." DO NOT USE ANY OTHER EXPRESSIONS OR ADD OTHER DETAILS.]
### [How many times this behavior occurs in OUTPUT A: "Count in OUTPUT A: {number}". DO NOT USE ANY OTHER EXPRESSIONS OR ADD OTHER DETAILS.]
### [Whether this reasoning behavior occurs in OUTPUT B: Either "This reasoning behavior is observed in OUTPUT B." or "This reasoning behavior is not observed in OUTPUT B." DO NOT USE ANY OTHER EXPRESSIONS OR ADD OTHER DETAILS.]
### [How many times this behavior occurs in OUTPUT B: "Count in OUTPUT B: {number}". DO NOT USE ANY OTHER EXPRESSIONS OR ADD OTHER DETAILS.]
### [Because of this reasoning behavior, which output is likely generated by which model]

...{Repeat for the rest of existing reasoning behaviors in the reasoning taxonomy}...

{If you observe the distinguishing reasoning behaviors that are not in the reasoning taxonomy}
{Add new distinguishing reasoning behaviors}
### [New distinguishing reasoning behavior's name]
### [Definition of this reasoning behavior (reasoning behavior); a quote or detailed summarization of this behavior]
### [Whether this new reasoning behavior occurs in OUTPUT A: Either "This reasoning behavior is observed in OUTPUT A." or "This reasoning behavior is not observed in OUTPUT A." DO NOT USE ANY OTHER EXPRESSIONS OR ADD OTHER DETAILS.]
### [How many times this behavior occurs in OUTPUT A: "Count in OUTPUT A: {number}". DO NOT USE ANY OTHER EXPRESSIONS OR ADD OTHER DETAILS.]
### [Whether this new reasoning behavior occurs in OUTPUT B: Either "This reasoning behavior is observed in OUTPUT B." or "This reasoning behavior is not observed in OUTPUT B." DO NOT USE ANY OTHER EXPRESSIONS OR ADD OTHER DETAILS.]
### [How many times this behavior occurs in OUTPUT B: "Count in OUTPUT B: {number}". DO NOT USE ANY OTHER EXPRESSIONS OR ADD OTHER DETAILS.]
### [Is this reasoning behavior and its definition really different from the existing reasoning behavior above? If not, then you shouldn't include this reasoning behavior in the Added reasoning behavior section of your final output!]
...{Repeat for the other new reasoning behaviors}...

[Which new distinguishing reasoning behavior is truly different from any of the existing ones in the system message. Again, you don't need to add a behavior unless it's truly different from known ones. Think step by step.]

Final output:
Added:
[Added distinguishing reasoning behavior name]: [Detailed reasoning behavior definition (reasoning behavior)] [Example of this behavior quoted from the given outputs or a detailed summarization of this behavior]]
------

Make sure you follow the exact format above when giving the added reasoning behavior. Write the reasoning behavior name, reasoning behavior definition, and example in the same line.

For the added reasoning behavior, think creatively. The added reasoning behaviors must separate two given outputs---that is it must occur significantly more in one of the outputs or only occur in one reasoning output. For example, it occurs in one of the outputs 7 times but only 3 times in other output. Or, it occurs in one of the reasoning outputs 1 time but not at all in other output.

Moreover, it should be different from the existing ones. Do not add reasoning behaviors that are similar to the existing ones in the reasoning taxonomy below in your Final output.

Below is the reasoning taxonomy you could use for the classification,
"""