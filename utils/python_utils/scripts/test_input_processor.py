import python_utils.input_file_processor as ifp

def main():
    file_processor = ifp.InputFileProcessor("test-file.zip")
    file_processor.process_input()

if __name__ == "__main__":
    main()