import pandas as pd

def split_data(threshold, entity):
    assignments = pd.read_csv(f"graphpart_assignments_{threshold}.csv")
    datasets = []

    for i in range(5):
        partitions = [0,1,2,3,4]
        partitions.pop(i)

        print(i)
        if i != 4:
            val_ids = assignments[assignments["cluster"] == partitions[i]]
            val_ids = val_ids["AC"]
            print(partitions[i])
            partitions.pop(i)
            
        else:
            val_ids = assignments[assignments["cluster"] == 0]
            val_ids = val_ids["AC"]
            partitions.pop(0)
            print(0)
        
        print(partitions)

        train_ids = assignments[assignments["cluster"].isin(partitions)]
        train_ids = train_ids["AC"]
        test_ids = assignments[assignments["cluster"] == i]
        test_ids = test_ids["AC"]
        datasets.append((train_ids, val_ids, test_ids))

        print("--------------------------------")

    from plants_sm.io.pickle import write_pickle

    write_pickle(f"splits_{threshold}_{entity}.pkl", datasets)

split_data("0_6", "proteins")
split_data("0_4", "proteins")

    
