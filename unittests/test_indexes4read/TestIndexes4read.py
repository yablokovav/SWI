import numpy as np
import unittest

from src.files_processor.readers import get_indexes4read

from src import root_dir

HEADER_FILE_PATH_FFID = root_dir / "unittests/data_csp_test/ffid.txt"
DARA_FILE_PATH = root_dir / "unittests/data/ffid_1_4.sgy"


def are_lists_equal(list1, list2):
    """Проверяет, что списки имеют одинаковую длину и соответствующие элементы равны."""
    if not isinstance(list1, list) or not isinstance(list2, list):
        print("Checked object is not list")
        return False
    if len(list1) != len(list2):
        print("Incorrect len of indexes list")
        return False
    for i in range(len(list1)):
        if not isinstance(list1[i], np.ndarray) or not isinstance(list2[i], np.ndarray):
            print("Incorrect format of indexes list")
            return False  # Если не NumPy массивы - считаем, что списки не равны
        if not np.array_equal(list1[i], list2[i]):
            print("Incorrect indexes values")
            return False
    return True


def read_indexes_from_two_functions(test_class,
                                    ranges,
                                    type_data,
                                    sort_3d_order,
                                    num_sources_on_cpu,
                                    haeder_file_path):
    try:
        indexes4read = test_class.get_indexes4read(
            haeder_file_path,
            DARA_FILE_PATH,
            ranges=ranges,
            type_data=type_data,
            sort_3d_order=sort_3d_order,
            num_sources_on_cpu=num_sources_on_cpu,
        )
    except ValueError as e:
        print(e)
        indexes4read = False

    try:
        indexes4read_base = test_class.get_indexes4read_fata_base(
            haeder_file_path,
            DARA_FILE_PATH,
            ranges=ranges,
            type_data=type_data,
            sort_3d_order=sort_3d_order,
            num_sources_on_cpu=num_sources_on_cpu,
        )
    except ValueError as e:
        print(e)
        indexes4read_base = False

    return indexes4read, indexes4read_base


class TestIndexes4read(unittest.TestCase):
    def setUp(self):
        self.get_indexes4read = get_indexes4read
        self.get_indexes4read_fata_base = get_indexes4read


    def testWithoutParallelFFID(self):
        ranges = (1, 4, 1)
        type_data = "3d"
        sort_3d_order = "csp"
        bin_size = (500, 500)
        num_sources_on_cpu = 0

        indexes4read, indexes4read_base = read_indexes_from_two_functions(
            self,
            ranges=ranges,
            type_data=type_data,
            sort_3d_order=sort_3d_order,
            num_sources_on_cpu=num_sources_on_cpu,
            haeder_file_path=HEADER_FILE_PATH_FFID,
        )
        if not indexes4read or not indexes4read_base:
            self.assertEqual(indexes4read, indexes4read_base)
        else:
            self.assertEqual(are_lists_equal(indexes4read, indexes4read_base), True,
                             'testWithoutParallelFFID Failed')
        print("testWithoutParallelFFID done")

    def testWithParallelFFID(self):
        ranges = (1, 4, 1)
        type_data = "3d"
        sort_3d_order = "csp"
        bin_size = (150, 150)
        num_sources_on_cpu = 2

        indexes4read, indexes4read_base = read_indexes_from_two_functions(
            self,
            ranges=ranges,
            type_data=type_data,
            sort_3d_order=sort_3d_order,
            num_sources_on_cpu=num_sources_on_cpu,
            haeder_file_path=HEADER_FILE_PATH_FFID,
        )

        if not indexes4read or not indexes4read_base:
            self.assertEqual(indexes4read, indexes4read_base)
        else:
            self.assertEqual(are_lists_equal(indexes4read, indexes4read_base), True,
                             "testWithParallelFFID Failed")
        print("testWithParallelFFID done")


    def testWithIncrementFFID(self):
        ranges = (1, 4, 2)
        type_data = "3d"
        sort_3d_order = "csp"
        bin_size = (150, 150)
        num_sources_on_cpu = 0

        indexes4read, indexes4read_base = read_indexes_from_two_functions(
            self,
            ranges=ranges,
            type_data=type_data,
            sort_3d_order=sort_3d_order,
            num_sources_on_cpu=num_sources_on_cpu,
            haeder_file_path=HEADER_FILE_PATH_FFID,
        )

        if not indexes4read or not indexes4read_base:
            self.assertEqual(indexes4read, indexes4read_base)
        else:
            self.assertEqual(are_lists_equal(indexes4read, indexes4read_base), True,
                             "testWithIncrementFFID Failed")
        print("testWithIncrementFFID done")


    def testWithIncrementAndParallelFFID(self):
        ranges = (1, 4, 2)
        type_data = "3d"
        sort_3d_order = "csp"
        bin_size = (150, 150)
        num_sources_on_cpu = 1

        indexes4read, indexes4read_base = read_indexes_from_two_functions(
            self,
            ranges=ranges,
            type_data=type_data,
            sort_3d_order=sort_3d_order,
            num_sources_on_cpu=num_sources_on_cpu,
            haeder_file_path=HEADER_FILE_PATH_FFID,
        )

        if not indexes4read or not indexes4read_base:
            self.assertEqual(indexes4read, indexes4read_base)
        else:
            self.assertEqual(are_lists_equal(indexes4read, indexes4read_base), True,
                             "testWithIncrementAndParallelFFID Failed")
        print("testWithIncrementAndParallelFFID done")

    def testWithStartStoplFFID(self):
        ranges = (2, 3, 1)
        type_data = "3d"
        sort_3d_order = "csp"
        bin_size = (150, 150)
        num_sources_on_cpu = 0

        indexes4read, indexes4read_base = read_indexes_from_two_functions(
            self,
            ranges=ranges,
            type_data=type_data,
            sort_3d_order=sort_3d_order,
            num_sources_on_cpu=num_sources_on_cpu,
            haeder_file_path=HEADER_FILE_PATH_FFID,
        )

        if not indexes4read or not indexes4read_base:
            self.assertEqual(indexes4read, indexes4read_base)
        else:
            self.assertEqual(are_lists_equal(indexes4read, indexes4read_base), True,
                             "testWithStartStoplFFID Failed")
        print("testWithStartStoplFFID done")

    def testWithStartStopIncrementFFID(self):
        ranges = (2, 3, 2)
        type_data = "3d"
        sort_3d_order = "csp"
        bin_size = (150, 150)
        num_sources_on_cpu = 0

        indexes4read, indexes4read_base = read_indexes_from_two_functions(
            self,
            ranges=ranges,
            type_data=type_data,
            sort_3d_order=sort_3d_order,
            num_sources_on_cpu=num_sources_on_cpu,
            haeder_file_path=HEADER_FILE_PATH_FFID,
        )

        if not indexes4read or not indexes4read_base:
            self.assertEqual(indexes4read, indexes4read_base)
        else:
            self.assertEqual(are_lists_equal(indexes4read, indexes4read_base), True,
                             "testWithStartStopIncrementFFID Failed")
        print("testWithStartStopIncrementFFID done")



if __name__ == '__main__':
    unittest.main()