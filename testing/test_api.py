import requests
import time

def test_race_data_api():
    # Base URL of the API
    base_url = "http://localhost:8000"
    
    # Test cases
    # {"year": 2024, "race": "jeddah"},
    test_cases = [
        {"year": 2024, "race": "singapore"},
    ]
    
    for test in test_cases:
        print(f"\nTesting {test['race'].title()} {test['year']} Grand Prix...")
        
        # Make the API request
        start_time = time.time()
        response = requests.get(f"{base_url}/race/{test['year']}/{test['race']}")
        end_time = time.time()
        
        # Print results
        print(f"Status Code: {response.status_code}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            # Save the response content to a file
            filename = f"test_output_{test['race']}_{test['year']}.txt"
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"Response saved to {filename}")
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    print("Starting API tests...")
    print("Make sure the API server is running (python extract_race_data.py)")
    print("Press Enter to continue...")
    input()
    
    try:
        test_race_data_api()
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")