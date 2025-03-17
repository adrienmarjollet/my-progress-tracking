import { Prompt, PromptChain, LlmPromptExecutor } from '../utils/prompt-chain';

/**
 * Example usage of the PromptChain class
 */
async function runExamplePromptChain() {
  // Create an executor for our prompts
  const executor = new LlmPromptExecutor('gpt-4');
  
  // Create some prompts
  const summaryPrompt = new Prompt(
    "Summarize the following text in 3 sentences: {{text}}",
    executor
  );
  
  const keyPointsPrompt = new Prompt(
    "Extract 5 key points from this summary: {{summary}}",
    executor
  );
  
  const actionItemsPrompt = new Prompt(
    "Based on these key points: {{keyPoints}}, suggest 3 action items.",
    executor
  );
  
  // Create a chain
  const chain = new PromptChain();
  
  // Add prompts to the chain
  chain.addPrompt(summaryPrompt, 'summary')
       .addPrompt(keyPointsPrompt, 'keyPoints', (outputs) => {
         // Map the 'summary' output to the 'summary' input expected by keyPointsPrompt
         return { summary: outputs.summary };
       })
       .addPrompt(actionItemsPrompt, 'actionItems', (outputs) => {
         // Map the 'keyPoints' output to the 'keyPoints' input expected by actionItemsPrompt
         return { keyPoints: outputs.keyPoints };
       });
  
  // Execute the chain with some initial text
  const result = await chain.execute({
    text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
  });
  
  // Output the results
  console.log("Summary:", result.summary);
  console.log("Key Points:", result.keyPoints);
  console.log("Action Items:", result.actionItems);
}

// Run the example
runExamplePromptChain().catch(console.error);
