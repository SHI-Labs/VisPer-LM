from ola_vlm.train.probe_dsg_train import train

if __name__ == "__main__":
    try:
        train(attn_implementation="flash_attention_2")
    except:
        train(attn_implementation="eager")