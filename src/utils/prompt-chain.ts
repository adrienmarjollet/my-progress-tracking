/**
 * Prompt Chaining Implementation
 * This module provides classes for creating chains of prompts that can be executed sequentially,
 * with each prompt's output potentially serving as input to subsequent prompts.
 */

interface PromptExecutor {
  execute(input: string): Promise<string>;
}

/**
 * Represents a single prompt in a chain
 */
class Prompt {
  private template: string;
  private executor: PromptExecutor;
  
  constructor(template: string, executor: PromptExecutor) {
    this.template = template;
    this.executor = executor;
  }

  /**
   * Formats the prompt template with the given variables
   */
  format(variables: Record<string, any>): string {
    let formatted = this.template;
    
    for (const [key, value] of Object.entries(variables)) {
      const placeholder = `{{${key}}}`;
      formatted = formatted.replace(new RegExp(placeholder, 'g'), String(value));
    }
    
    return formatted;
  }

  /**
   * Executes the prompt with the given variables
   */
  async execute(variables: Record<string, any> = {}): Promise<string> {
    const formattedPrompt = this.format(variables);
    return this.executor.execute(formattedPrompt);
  }
}

/**
 * A chain of prompts that can be executed sequentially
 */
class PromptChain {
  private prompts: Array<{
    prompt: Prompt,
    outputKey: string,
    inputMapper?: (prevOutputs: Record<string, any>) => Record<string, any>
  }> = [];
  
  /**
   * Adds a prompt to the chain
   * @param prompt The prompt to add
   * @param outputKey The key to store the output under
   * @param inputMapper Optional function to map previous outputs to this prompt's inputs
   */
  addPrompt(
    prompt: Prompt, 
    outputKey: string,
    inputMapper?: (prevOutputs: Record<string, any>) => Record<string, any>
  ): this {
    this.prompts.push({ prompt, outputKey, inputMapper });
    return this;
  }
  
  /**
   * Executes the entire prompt chain
   * @param initialVariables Initial variables to start the chain with
   */
  async execute(initialVariables: Record<string, any> = {}): Promise<Record<string, any>> {
    const outputs: Record<string, any> = { ...initialVariables };
    
    for (const { prompt, outputKey, inputMapper } of this.prompts) {
      const inputVariables = inputMapper ? inputMapper(outputs) : outputs;
      const result = await prompt.execute(inputVariables);
      outputs[outputKey] = result;
    }
    
    return outputs;
  }
}

/**
 * Example implementation of a prompt executor that uses a hypothetical LLM API
 */
class LlmPromptExecutor implements PromptExecutor {
  private model: string;
  
  constructor(model: string = 'default-model') {
    this.model = model;
  }
  
  async execute(input: string): Promise<string> {
    // In a real implementation, this would call an actual LLM API
    console.log(`Executing prompt with model ${this.model}: "${input.substring(0, 50)}..."`);
    
    // Simulating API call delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    return `This is a response for: ${input.substring(0, 30)}...`;
  }
}

// Export all classes
export {
  Prompt,
  PromptChain,
  PromptExecutor,
  LlmPromptExecutor
};
