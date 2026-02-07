from .validator import ValidationSuite


def main():
    suite = ValidationSuite()
    suite.run_full_validation()


if __name__ == "__main__":
    main()
