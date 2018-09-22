if __name__ == "__main__":

    dimension = 10
    streak = 5
    filename = "examples/board_"+str(dimension)+"_"+str(streak)+".txt"
    row = 1
    col = 1
    winning_paths = []

    #check horizontal
    for row in range(1,dimension+1):
        for col in range(1, dimension + 1):
            i = (row-1)*dimension+col
            if (i+(streak-1))<=(dimension*row): #horizontal paths
                path = []
                for s in range(0,streak):
                    path.append(i+s)
                winning_paths.append(path)

            if (i+(streak-1)*dimension)<=dimension*(dimension-1)+col: #vertical paths
                path = []
                for s in range(0,streak):
                    path.append(i+(s)*dimension)
                winning_paths.append(path)

            if (i+(streak-1)*(dimension+1))<=dimension*dimension: #diagonal right paths
                if (i + (streak - 1) * (dimension + 1)) <= (row + (streak - 1)) * dimension:  # diagonal right paths
                    path = []
                    for s in range(0,streak):
                        path.append(i+(s)*(dimension+1))
                    winning_paths.append(path)

            if (i+(streak-1)*(dimension-1))<=dimension*dimension: #diagonal right paths
                if (i + (streak - 1) * (dimension - 1)) > ((row-1) + (streak - 1)) * dimension:  # diagonal right paths
                    path = []
                    for s in range(0,streak):
                        path.append(i+(s)*(dimension-1))
                    winning_paths.append(path)

    with open(filename, "w") as text_file:
        text_file.write(str(dimension*dimension))
        text_file.write("\n")
        for path in winning_paths:
            for i in range(len(path)):
                text_file.write(str(path[i]))
                if i<len(path)-1:
                    text_file.write(" ")
            text_file.write("\n")
    print winning_paths



