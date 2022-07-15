import zipfile

def main():
    with zipfile.ZipFile("test-file.zip","w") as zip_file:
        with open("testfile.txt","w") as test_file:
            test_file.write("Hello, World!")
        zip_file.write("testfile.txt","testfile.txt")

if __name__ == "__main__":
    main()