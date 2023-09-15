# Malachi Eberly
# Assignment 1: DP Problem

OUTPUT = 0

def waterTrapped(input):
    global OUTPUT

    # If we finished the last level, we can return out counted blocks of water
    if max(input) == 0:
        return OUTPUT
    
    # Remove leading zeros
    while input and input[0] == 0:
        input.pop(0)
    # Remove trailing zeros
    while input and input[-1] == 0:
        input.pop()
    # Add the number of remaining zeros to total
    for j in range(len(input)):
        if input[j] == 0:
            OUTPUT += 1
        else:
            input[j] -= 1

    return waterTrapped(input)

def main():
    input = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    print("The example problem can hold", waterTrapped(input), "blocks of water")

if __name__ == "__main__":
    main()