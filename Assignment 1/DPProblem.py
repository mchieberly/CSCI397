# Malachi Eberly
# Assignment 1: DP Problem

def waterTrapped(input):
    maxHeight = max(input)
    output = 0
    currentList = input
    for i in range(maxHeight):
        # Remove leading zeros
        while currentList and currentList[0] == 0:
            currentList.pop(0)
        # Remove trailing zeros
        while currentList and currentList[-1] == 0:
            currentList.pop()
        # Add the number of remaining zeros to total
        for j in range(len(currentList)):
            if currentList[j] == 0:
                output += 1
            else:
                currentList[j] -= 1
    return output

def main():
    input = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    print("The example problem can hold", waterTrapped(input), "blocks of water")

if __name__ == "__main__":
    main()