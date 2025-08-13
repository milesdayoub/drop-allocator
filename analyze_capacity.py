#!/usr/bin/env python3
"""
Sponsored Contract Capacity Analysis
Run this to see how much sponsored capacity is being used vs available
"""

import pandas as pd

def analyze_sponsored_capacity():
    print("=== SPONSORED CONTRACT CAPACITY ANALYSIS ===\n")
    
    # Load the data
    print("Loading data...")
    caps = pd.read_csv('data/clean_contract_caps.csv')
    assignments = pd.read_csv('outputs/assignments.csv')
    
    # Filter sponsored contracts
    sponsored_caps = caps[caps['is_sponsored'] == True]
    total_sponsored_cap = sponsored_caps['cap_face'].sum()
    
    print(f"Total sponsored contract capacity: {total_sponsored_cap:,}")
    
    # Count sponsored assignments
    sponsored_assignments = assignments[assignments['is_sponsored'] == True]
    total_sponsored_assignments = len(sponsored_assignments)
    
    print(f"Total sponsored assignments made: {total_sponsored_assignments:,}")
    print(f"Overall sponsored capacity utilization: {total_sponsored_assignments/total_sponsored_cap*100:.1f}%\n")
    
    # Analyze by contract
    sponsored_usage = sponsored_assignments.groupby('contract_address').size()
    
    # Merge with caps
    capacity_analysis = sponsored_caps.merge(
        sponsored_usage.rename('assigned').reset_index(), 
        on='contract_address', 
        how='left'
    ).fillna(0)
    
    capacity_analysis['utilization_pct'] = (capacity_analysis['assigned'] / capacity_analysis['cap_face'] * 100).round(1)
    capacity_analysis['remaining'] = capacity_analysis['cap_face'] - capacity_analysis['assigned']
    
    # Show fully utilized contracts
    fully_utilized = capacity_analysis[capacity_analysis['assigned'] >= capacity_analysis['cap_face']]
    if not fully_utilized.empty:
        print(f"🚨 FULLY UTILIZED sponsored contracts ({len(fully_utilized)}):")
        print(fully_utilized[['contract_address', 'cap_face', 'assigned', 'utilization_pct']].head(10))
        print()
    
    # Show contracts with remaining capacity
    remaining_capacity = capacity_analysis[capacity_analysis['remaining'] > 0]
    if not remaining_capacity.empty:
        print(f"✅ Sponsored contracts with remaining capacity ({len(remaining_capacity)}):")
        print(remaining_capacity[['contract_address', 'cap_face', 'assigned', 'remaining', 'utilization_pct']].head(10))
        print()
    
    # Show unused contracts
    unused = capacity_analysis[capacity_analysis['assigned'] == 0]
    if not unused.empty:
        print(f"⚠️  Unused sponsored contracts ({len(unused)}):")
        print(unused[['contract_address', 'cap_face']].head(10))
        print()
    
    # Summary
    print("=== SUMMARY ===")
    print(f"Fully utilized: {len(fully_utilized)} contracts")
    print(f"Partially used: {len(remaining_capacity)} contracts") 
    print(f"Unused: {len(unused)} contracts")
    print(f"Total sponsored contracts: {len(sponsored_caps)}")
    
    if len(fully_utilized) > 0:
        print(f"\n🚨 CAPACITY CONSTRAINT: {len(fully_utilized)} sponsored contracts are at 100% capacity!")
        print("This explains why you can't meet the minimum sponsorship ratios.")
        print("Consider: reducing ratio targets, adding more sponsored contracts, or using soft constraints.")

if __name__ == "__main__":
    analyze_sponsored_capacity()
