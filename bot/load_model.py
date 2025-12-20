from src.models.llama_model import LlamaModel


class LlamaBot:
    def __init__(self,model_path:str="./checkpoints/reward_model"):
        self.model = LlamaModel(checkpoint_path=model_path, use_checkpoint=True )

    def answer(self, message:str):
        prompt = f"Your are helpfull assisstant. Answer to question : {message}"
        answer = self.model.generate_text(prompt=prompt)
        print(answer)
        return answer


if __name__ == "__main__":
    bot = LlamaBot()
    bot.answer("What is human feedback?")
    