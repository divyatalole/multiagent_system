#!/usr/bin/env python3
"""
Quick test script to verify all components are working
"""
import requests
import time
import sys

API_BASE = "http://localhost:8000"
STREAMLIT_BASE = "http://localhost:8501"

def test_api_health():
    """Test API health endpoint"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            print("[OK] API Health check passed")
            return True
        else:
            print(f"[FAIL] API Health returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] API Health check failed: {e}")
        return False

def test_api_agents():
    """Test API agents endpoint"""
    try:
        response = requests.get(f"{API_BASE}/agents", timeout=10)
        if response.status_code == 200:
            agents = response.json()
            print(f"[OK] API Agents endpoint works - Found {len(agents)} agents")
            return True
        else:
            print(f"[FAIL] API Agents returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] API Agents check failed: {e}")
        return False

def test_api_analyze():
    """Test API analyze endpoint"""
    try:
        payload = {"topic": "AI personal finance app for millennials", "max_results": 3}
        response = requests.post(f"{API_BASE}/analyze", json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                print("[OK] API Analyze endpoint works")
                return True
            else:
                print(f"[FAIL] API Analyze returned status: {data.get('status')}")
                return False
        else:
            print(f"[FAIL] API Analyze returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] API Analyze check failed: {e}")
        return False

def test_streamlit():
    """Test Streamlit UI"""
    try:
        response = requests.get(f"{STREAMLIT_BASE}/", timeout=10)
        if response.status_code == 200:
            print("[OK] Streamlit UI is accessible")
            return True
        else:
            print(f"[FAIL] Streamlit returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Streamlit check failed: {e}")
        return False

def main():
    print("=" * 60)
    print("StartupAI Project - Component Testing")
    print("=" * 60)
    print()
    
    print("Waiting 5 seconds for services to initialize...")
    time.sleep(5)
    print()
    
    results = []
    
    print("[1/4] Testing API Health...")
    results.append(test_api_health())
    print()
    
    print("[2/4] Testing API Agents...")
    results.append(test_api_agents())
    print()
    
    print("[3/4] Testing API Analyze...")
    results.append(test_api_analyze())
    print()
    
    print("[4/4] Testing Streamlit UI...")
    results.append(test_streamlit())
    print()
    
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print()
    
    if passed == total:
        print("[SUCCESS] All components are working!")
        print()
        print("Access points:")
        print(f"  - API: {API_BASE}")
        print(f"  - UI:  {STREAMLIT_BASE}")
        return 0
    else:
        print("[WARNING] Some components may not be working")
        return 1

if __name__ == "__main__":
    sys.exit(main())

