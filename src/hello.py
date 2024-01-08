import wandb

def main():
    wandb.init(project="my-first-sweep", group="my-first-group", job_type="my-first-job", save_code=True)
    wandb.log({"loss": 0.5})
        
if __name__ == "__main__":
    main()