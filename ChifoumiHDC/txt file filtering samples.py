import re

# extract and reformat valid file paths when i was making the picked samples from pkl.
def extract_paths(input_file, output_file):
    valid_paths = []

    path_pattern = re.compile(r"/space/.*?\.pkl")

    with open(input_file, "r") as file:
        for line in file:
            match = path_pattern.search(line)
            if match:
                valid_paths.append(match.group(0))
    with open(output_file, "w") as file:
        for path in valid_paths:
            file.write(path + "\n")

    print(f" extracted and saved {len(valid_paths)} paths to {output_file}")


if __name__ == "__main__":
    input_file= "/space/chair-nas/tosy/preprocessed_dat_chifoumi/picked samples"
    output_file = "/space/chair-nas/tosy/preprocessed_dat_chifoumi/filtered"
    extract_paths(input_file, output_file)
