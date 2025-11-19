"""
Test Suite for Data Pipeline
Kiểm tra tất cả components của pipeline
"""

import sys
from pathlib import Path
import unittest

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from csv_reader import (
    read_company_data_from_csv,
    validate_company_data,
    get_companies_summary
)


class TestCSVReader(unittest.TestCase):
    """Test CSV reader module"""
    
    def setUp(self):
        self.csv_path = './data/company_info.csv'
    
    def test_read_csv_success(self):
        """Test đọc CSV thành công"""
        df = read_company_data_from_csv(self.csv_path)
        self.assertFalse(df.empty, "DataFrame should not be empty")
        self.assertIn('symbol', df.columns, "Should have 'symbol' column")
        self.assertIn('organ_name', df.columns, "Should have 'organ_name' column")
    
    def test_validate_data(self):
        """Test validate dữ liệu"""
        df = read_company_data_from_csv(self.csv_path)
        if not df.empty:
            is_valid, errors = validate_company_data(df)
            self.assertTrue(is_valid or len(errors) > 0, "Should return validation result")
    
    def test_get_summary(self):
        """Test lấy summary"""
        df = read_company_data_from_csv(self.csv_path)
        if not df.empty:
            summary = get_companies_summary(df)
            self.assertIn('total_companies', summary, "Should have total_companies")
            self.assertIn('by_exchange', summary, "Should have by_exchange")
            self.assertGreater(summary['total_companies'], 0, "Should have at least 1 company")


class TestVNStockFetcher(unittest.TestCase):
    """Test VNStock fetcher module"""
    
    def test_import_module(self):
        """Test import module"""
        try:
            from vnstock_fetcher import (
                fetch_stock_data_from_vnstock,
                process_stock_data
            )
            self.assertTrue(True, "Module imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import module: {e}")


class TestPostgresConnector(unittest.TestCase):
    """Test Postgres connector module"""
    
    def test_import_module(self):
        """Test import module"""
        try:
            from postgres_connector import (
                setup_postgres_connection,
                create_database_schema
            )
            self.assertTrue(True, "Module imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import module: {e}")


class TestPipelineOrchestrator(unittest.TestCase):
    """Test Pipeline orchestrator"""
    
    def test_import_module(self):
        """Test import module"""
        try:
            from pipeline_orchestrator import (
                main_pipeline,
                run_pipeline
            )
            self.assertTrue(True, "Module imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import module: {e}")


def run_tests():
    """Chạy tất cả tests"""
    print("\n" + "="*60)
    print("RUNNING DATA PIPELINE TESTS")
    print("="*60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCSVReader))
    suite.addTests(loader.loadTestsFromTestCase(TestVNStockFetcher))
    suite.addTests(loader.loadTestsFromTestCase(TestPostgresConnector))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineOrchestrator))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
