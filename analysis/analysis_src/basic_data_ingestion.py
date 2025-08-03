from abc import ABC, abstractmethod
import pandas as pd

class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: This method prints the inspection results directly.
        """
        pass


class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())

class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print(df.describe())
        print(df.describe(include=["O"]))


class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def set_stratgey(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's inspection method.
        """
        self._strategy.inspect(df)


if __name__ == "__main__":
    # Example usage of the DataInspector with different strategies.

    # #Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # #Initialize the Data Inspector with a specific strategy
    # inspector = DataInspector(DataTypesInspectionStrategy())
    # inspector.execute_inspection(df)

    # #Change strategy to Summary Statistics and execute
    # inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    # inspector.execute_inspection(df)
    pass

