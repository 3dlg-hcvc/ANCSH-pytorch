from ANCSH_lib import ANCSHTrainer
from time import time

if __name__ == "__main__":
    start = time()
    trainer = ANCSHTrainer(data_path=None, max_epochs=1000)

    trainer.train()

    stop = time()
    print(str(stop - start) + " seconds")
