import os
import json
import subprocess
import shutil
import sys
import uuid
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


def run_command(command):
    """Run a shell command and check for errors."""
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error: Command failed with exit code {result.returncode}")
        sys.exit(1)
    print(f"Command completed successfully.")


def load_task_data(task_data_path):
    """Load task data from task_data.json."""
    with open(task_data_path, "r") as file:
        return json.load(file)


def load_implementation_reports(analysis_dir):
    """Load reports for all implementations."""
    reports = {}
    for i in range(1, 5):
        report_path = os.path.join(analysis_dir, f"impl_{i}_report.md")
        with open(report_path, "r") as file:
            reports[f"impl_{i}"] = file.read()
    return reports


def analyze_implementations(
    problem_description,
    reports,
    llm,
    implementations,
    include_improved_diff=False,
):
    system_message = """
    You are a software engineer specializing in analyzing and comparing code implementations for specific requirements. 
    Your primary task is to evaluate the provided code implementations, assign scores, and generate detailed justifications 
    based on the following format:

    EXPECTED OUTPUT FORMAT:
    {
      "implementations": [
        {
          "name": "impl_1",
          "task_success": 0,
          "instruction_following": 0,
          "comment": ""
        },
        {
          "name": "impl_2",
          "task_success": 0,
          "instruction_following": 0,
          "comment": ""
        },
        {
          "name": "impl_3",
          "task_success": 0,
          "instruction_following": 0,
          "comment": ""
        },
        {
          "name": "impl_4",
          "task_success": 0,
          "instruction_following": 0,
          "comment": ""
        }
      ],
      "evaluation": {
        "best_diff": 0,
        "preference_rationale": "",
        "improved_diff": ""
      }
    }

    SCORING CRITERIA:
    1. **Task Success (Rate 1-5)**:
       - Did the implementation complete the task successfully?
       - Did the resulting code work as intended?
       - Were unresolved issues present?
       - How well-designed was the code solution?
       - Did the implementation follow best practices?
       - Was the code clean, readable, and maintainable?
       - Did it integrate well with the existing codebase?
       - Were edge cases appropriately considered?

       **Format for Task Success Justification**:
       - General Outcome: Provide a brief description of what the implementation accomplished.
       - Conversation Goals:
         - Goal 1: Achieved — explain why.
         - Goal 2: Partially achieved — explain issues.
         - Goal 3: Not achieved — explain why.

    2. **Instruction Following (Rate 1-5)**:
       - Did the implementation follow the instructions precisely?
       - Did it avoid overreaching or making unnecessary assumptions?
       - Was it responsive to corrections or clarifications?

       **Format for Instruction Following Justification**:
       - General Outcome: Describe how well the instructions were followed.
       - Specific Evidence: Mention examples from the implementation.

    3. **Comment**:
       - Provide a specific, concise comment (min 90 words, max 160 words) about the implementation. Highlight strengths and weaknesses referencing the code implemented. Value example: "Good implementation of the retry logic with proper exponential backoff. However, the error messages are generic and don't include context like attempt number or delay time. Consider adding structured logging with error codes."
       - It's important and mandatory to reference or naming the core functions or updated files in your comments.
       - The explanation should be relative to all technical aspects of the implementation, not just the general outcome.

    EVALUATION REQUIREMENTS:
    1. **Best Diff**:
       - Identify the implementation that best satisfies the requirement. The value is the number of the chosen implementation (1, 2, 3, or 4).
    2. **Preference Rationale**:
       - min 250 words max 350 words.
       - Explain why the selected implementation is the best. Reference specific code decisions, patterns, and practices referencing the code implemented. Detailed comparative analysis explaining why the selected implementation is considered best, with specific technical details about differences between all four implementations, example: Implementation 2 is the best choice. Implementation 1 introduces an unnecessary 'timeout_seconds' parameter to the 'handle_request' function that isn't used anywhere in the code, whereas Implementation 2 keeps the function signature clean. Implementation 3 uses a basic try-catch block without proper retry logic, missing the exponential backoff requirement. Implementation 4 implements retry logic but uses a fixed 1-second delay instead of exponential backoff as specified. Implementation 2 correctly implements exponential backoff with configurable base delay, uses async/await properly for non-blocking operations, and includes comprehensive error logging. The 'calculate_delay' helper function in Implementation 2 is also more readable than the inline calculations in Implementation 4.
       - Examples of Strong Rationale Elements
        Architecture and Design Patterns: "Implementation 1 employs the Strategy pattern by creating a separate RetryStrategy interface with concrete implementations (ExponentialBackoffStrategy, LinearRetryStrategy). This decouples the retry algorithm from the client code, making it more maintainable as new strategies can be added without modifying existing code. In contrast, Implementation 2 hardcodes the retry logic directly in the request handler, violating the Open/Closed Principle."
        Error Handling and Robustness: "Implementation 3 handles edge cases comprehensively: it validates input parameters (checking for negative retry counts), catches specific exception types (TimeoutError, ConnectionError, HTTPError) with appropriate responses, and implements proper cleanup in finally blocks. Implementation 4 uses a generic except Exception block which can mask important errors and doesn't handle resource cleanup, potentially leading to memory leaks in long-running applications."
        Performance and Scalability: "Implementation 2 uses async/await with asyncio.gather() for concurrent requests, reducing total execution time from O(n) to O(1) for batch operations. It also implements connection pooling with a maximum of 100 connections, preventing resource exhaustion. Implementation 1 processes requests sequentially and creates new connections for each request, which will not scale effectively under high load and may exhaust system resources."
        Code Organization and Maintainability: "Implementation 4 separates concerns effectively: configuration is isolated in a Config class, business logic is in dedicated service classes, and HTTP handling is in controller classes. This follows the Single Responsibility Principle and makes testing easier. Implementation 3 mixes configuration, business logic, and HTTP handling in a single file, making it difficult to unit test individual components and harder to modify without affecting unrelated functionality."
        Testing and Verification: "Implementation 1 includes comprehensive test coverage with unit tests for each component (95% coverage), integration tests for the full workflow, and property-based tests using Hypothesis to verify edge cases. It also includes performance benchmarks and load tests. Implementation 2 only includes basic happy-path tests (60% coverage) and lacks integration tests, making it risky to deploy in production environments."
        Resource Management and Memory Efficiency: "Implementation 3 properly manages database connections using context managers and connection pooling, ensuring connections are released after use and preventing connection leaks. The implementation also uses lazy loading for large datasets, reducing memory footprint. Implementation 1 creates new database connections for each operation without proper cleanup, leading to connection exhaustion under moderate load."
        Security and Input Validation: "Implementation 2 includes input sanitization using parameterized queries and validates all user inputs against a whitelist, preventing SQL injection and XSS attacks. It also implements rate limiting and authentication checks. Implementation 4 uses string concatenation for SQL queries and lacks input validation, creating significant security vulnerabilities that could be exploited in production."
        Always mention all the implementations in the evaluation, even if one is clearly superior.

    3. **Improved Diff**:
       -  Optional improved implementation, generate only if include_improved_diff == True.
       -  To generate the improved implementation should combines the best aspects of given solutions or addresses their shortcomings, if applicable. The Improved should be a unified diff text, The improvement should take as a model the diff(n) value according to the best implementation according to the analysis, and add or remove the necessary code to improve the implementation.
       -  ALWAYS add short comments explaining the improvement and return the updated diff with the improvements. 

    OUTPUT STRICTLY IN JSON FORMAT. Ensure all descriptions align with the scores provided.
    """
    # Combine problem description and implementation details
    user_prompt = f"""
    TASK DESCRIPTION:
    {problem_description}

    INCLUDE IMPROVED DIFF: {include_improved_diff}

    IMPLEMENTATION DETAILS:
    Implementation 1:
    {reports['impl_1']}
    diff_1: {implementations[0]}

    Implementation 2:
    {reports['impl_2']}
    diff_2: {implementations[1]}

    Implementation 3:
    {reports['impl_3']}
    diff_3: {implementations[2]}

    Implementation 4:
    {reports['impl_4']}
    diff_4: {implementations[3]}
    """

    # Send the request to the LLM
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)

    try:
        content = response.content
        first_brace = content.find("{")
        last_brace = content.rfind("}")

        if first_brace != -1 and last_brace != -1:
            json_str = content[first_brace : last_brace + 1]
            return json.loads(json_str)
        else:
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Raw response content: {content}")
        return {
            "implementations": [
                {
                    "name": "impl_1",
                    "task_success": 0,
                    "instruction_following": 0,
                    "comment": "Error parsing LLM response",
                },
                {
                    "name": "impl_2",
                    "task_success": 0,
                    "instruction_following": 0,
                    "comment": "Error parsing LLM response",
                },
                {
                    "name": "impl_3",
                    "task_success": 0,
                    "instruction_following": 0,
                    "comment": "Error parsing LLM response",
                },
                {
                    "name": "impl_4",
                    "task_success": 0,
                    "instruction_following": 0,
                    "comment": "Error parsing LLM response",
                },
            ],
            "evaluation": {
                "best_diff": 0,
                "preference_rationale": "Error parsing LLM response",
                "improved_diff": "",
            },
        }


def update_metadata(task_folder, analysis_results):
    """Update metadata.json by iterating over the results without overwriting existing data."""
    metadata_path = os.path.join(task_folder, "metadata.json")
    with open(metadata_path, "r") as file:
        metadata = json.load(file)

    # Create a mapping between implementation names and their corresponding diff keys
    impl_to_diff = {
        "impl_1": "diff_1",
        "impl_2": "diff_2",
        "impl_3": "diff_3",
        "impl_4": "diff_4",
    }

    # Update implementations section
    for implementation in analysis_results["implementations"]:
        impl_name = implementation["name"]
        task_success = implementation["task_success"]
        instruction_following = implementation["instruction_following"]
        comment = implementation["comment"]

        # Get the corresponding diff key for this implementation
        diff_key = impl_to_diff.get(impl_name)
        if not diff_key:
            continue  # Skip if no matching diff key

        # Find and update the matching implementation in metadata
        for impl in metadata["implementations"]:
            if (
                diff_key in impl
            ):  # Check if this metadata entry has the matching diff key
                impl["task_success"] = task_success
                impl["instruction_following"] = instruction_following
                impl["comment"] = comment
                break  # Found and updated the matching entry

    # Update evaluation section
    if "evaluation" in analysis_results:
        if "best_diff" in analysis_results["evaluation"]:
            metadata["evaluation"]["best_diff"] = analysis_results["evaluation"][
                "best_diff"
            ]
        if "preference_rationale" in analysis_results["evaluation"]:
            metadata["evaluation"]["preference_rationale"] = analysis_results[
                "evaluation"
            ]["preference_rationale"]
        if "improved_diff" in analysis_results["evaluation"]:
            metadata["evaluation"]["improved_diff"] = analysis_results["evaluation"][
                "improved_diff"
            ]

    # Write the updated metadata back to the file
    with open(metadata_path, "w") as file:
        json.dump(metadata, file, indent=4)


def create_task_folder_and_metadata(task_data_path):
    if not os.path.exists(task_data_path):
        print(f"Error: {task_data_path} not found.")
        sys.exit(1)

    # Load task_data.json
    with open(task_data_path, "r") as task_data_file:
        task_data = json.load(task_data_file)

    # Extract jira_id for the folder name
    jira_id = task_data.get("jira_id")
    if not jira_id:
        print("Error: jira_id not found in task_data.json")
        sys.exit(1)

    # Define the new task folder path
    task_folder = os.path.join("tasks", jira_id)
    os.makedirs(task_folder, exist_ok=True)
    print(f"Created folder: {task_folder}")

    # Copy metadata_base.json to the new folder
    shutil.copyfile(
        "tasks/metadata_base.json", os.path.join(task_folder, "metadata.json")
    )
    print(f"Copied metadata_base.json to {task_folder}")

    # Update metadata.json with task_data.json values
    metadata_path = os.path.join(task_folder, "metadata.json")
    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)

    # Generate UUID4 for task_id
    task_id = str(uuid.uuid4())

    # Update fields in metadata.json
    metadata["task_id"] = task_id  # Add the generated UUID4
    metadata["jira"] = task_data.get("jira_id", "")
    metadata["task_details"]["repo_name"] = task_data.get("repo_name", "")
    metadata["task_details"]["before_sha"] = task_data.get("before_sha", "")
    metadata["task_details"]["after_sha"] = task_data.get("after_sha", "")
    metadata["task_details"]["pr_description"] = task_data.get("pr_description", "")
    metadata["task_details"]["reference_implementation_diff"] = task_data.get(
        "reference_implementation_diff", ""
    )
    metadata["implementations"][0]["diff_1"] = task_data.get("diff_1", "")
    metadata["implementations"][1]["diff_2"] = task_data.get("diff_2", "")
    metadata["implementations"][2]["diff_3"] = task_data.get("diff_3", "")
    metadata["implementations"][3]["diff_4"] = task_data.get("diff_4", "")

    # Save the updated metadata.json
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=4)
    print(f"Updated metadata.json in {task_folder} with task_id: {task_id}")

    return task_folder


def main():
    # Paths and setup
    excecute_commands = True
    task_data_path = os.path.join("automation_tool", "task_data.json")
    analysis_dir = os.path.join("code-evaluation", "analysis")
    include_improved_diff = False

    # Initialize LLM
    api_key
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    # Step 1: Execute commands to generate reports
    commands = [
        "python automation_tool/code_implementation_analysis.py automation_tool/task_data.json",
        "python automation_tool/compare_implementations.py --dir code-evaluation report impl_1 --output code-evaluation/analysis/impl_1_report.md",
        "python automation_tool/compare_implementations.py --dir code-evaluation report impl_2 --output code-evaluation/analysis/impl_2_report.md",
        "python automation_tool/compare_implementations.py --dir code-evaluation report impl_3 --output code-evaluation/analysis/impl_3_report.md",
        "python automation_tool/compare_implementations.py --dir code-evaluation report impl_4 --output code-evaluation/analysis/impl_4_report.md",
        "python automation_tool/compare_implementations.py --dir code-evaluation matrix --output code-evaluation/analysis/matrix.md",
    ]

    if excecute_commands:
        for command in commands:
            run_command(command)

    # Step 2: Create task folder and load data
    task_folder = create_task_folder_and_metadata(task_data_path)
    task_data = load_task_data(task_data_path)
    problem_description = task_data["pr_description"]
    implementations = [
        task_data["diff_1"],
        task_data["diff_2"],
        task_data["diff_3"],
        task_data["diff_4"],
    ]
    reports = load_implementation_reports(analysis_dir)

    print("Analysis reports in progress!")
    # Step 3: Analyze implementations using LangChain
    analysis_results = analyze_implementations(
        problem_description, reports, llm, implementations, include_improved_diff
    )

    # Step 4: Update metadata.json with analysis results
    update_metadata(task_folder, analysis_results)
    print("Metadata updated successfully!")


if __name__ == "__main__":
    main()