from ola_vlm.train.sherlock_train import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
    # train(attn_implementation="eager")
