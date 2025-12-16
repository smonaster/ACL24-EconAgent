import simulate

def run():
    simulate.main(
        policy_model='gpt',
        num_agents=12,
        episode_length=2,
        dialog_len=3,
        # añade otros parámetros si los necesitas
    )

if __name__ == "__main__":
    run()

