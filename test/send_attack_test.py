# test/send_attack_test.py
from kafka import KafkaProducer
import json
import time
import random
import logging

logging.basicConfig(level=logging.INFO)

def main():
    producer = KafkaProducer(
        bootstrap_servers='127.0.0.1:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    attack_types = [
        # Normal Traffic (High frequency to simulate realistic ratios)
        ("Benign", 80), ("Benign", 443), ("Benign", 22), ("Benign", 53),
        
        # Ransomware & Extortion
        ("Ransomware-WannaCry", 445), ("Ransomware-Ryuk", 3389), ("Ransomware-Conti", 443),
        
        # Physical / Network Level Disruptions
        ("DDoS-SynFlood", 80), ("DDoS-UDPFlood", 53), ("DDoS-PingOfDeath", 0),
        ("DDoS-Smurf", 0), ("DDoS-LOIC-HTTP", 80), ("DDoS-HOIC", 443),
        ("BGP-Hijacking", 179), ("ManInTheMiddle-ARP", 0), ("DNS-Spoofing", 53),
        
        # Botnets
        ("Botnet-Mirai", 23), ("Botnet-Qbot", 443), ("Botnet-Emotet", 8080),
        ("Botnet-TrickBot", 443), ("Botnet-Andromeda", 0),
        
        # Web Vulnerabilities & Injections
        ("SQLi-Union", 443), ("SQLi-ErrorBased", 80), ("SQLi-Blind", 443),
        ("CrossSiteScripting-Reflected", 80), ("CrossSiteScripting-Stored", 443),
        ("CommandInjection-OS", 22), ("DirectoryTraversal", 80),
        ("InsecureDeserialization", 8080), ("WebShell-Upload", 443),
        
        # Zero-Day Exploits & CVEs
        ("ZeroDay-Log4j", 8080), ("ZeroDay-PrintNightmare", 135), 
        ("ZeroDay-ProxyLogon", 443), ("ZeroDay-Spring4Shell", 8080),
        ("Exploit-EternalBlue", 445), ("Exploit-Heartbleed", 443), 
        ("Exploit-Shellshock", 80),
        
        # Brute Force & Credential Attacks
        ("CredentialStuffing-API", 443), ("BruteForce-SSH", 22), 
        ("BruteForce-RDP", 3389), ("BruteForce-FTP", 21), ("BruteForce-Smb", 445),
        
        # Advanced Persistent Threats (APTs)
        ("APT-Lazarus", 443), ("APT-CozyBear", 8080), ("APT-FancyBear", 443),
        ("Malware-CobaltStrike", 4444), ("C2-Beacon", 443), ("C2-DNS-Tunnel", 53),
        ("DataExfiltration-DNS", 53), ("DataExfiltration-HTTPS", 443),
        
        # Next-Gen Vectors
        ("Cryptojacking-XMRig", 3333), ("API-BrokenAuth", 443), 
        ("API-RateLimitBypass", 80), ("GraphQL-IntrospectionLeak", 443), 
        ("SSRF-CloudMetaData", 80), ("IoT-CamHack", 81), ("IoT-RouterExploit", 8080)
    ]

    print("🚀 Firing Real-Time Attack Streams to Kafka (127.0.0.1:9092) - Press Ctrl+C to stop")
    
    try:
        count = 0
        while True:
            attack, port = random.choice(attack_types)
            msg = {
                "Dst Port": port,
                "Protocol": 6,
                "Label": attack,
                # Add dummy variables required by the feature array size
                "Flow Duration": random.randint(100, 10000),
                "Tot Fwd Pkts": random.randint(1, 50),
                "timestamp": time.time()
            }
            producer.send('network_traffic', msg)
            count += 1
            if count % 10 == 0:
                print(f"[{count}] Transmitted 10 packets... (Example: {attack})")
                producer.flush()
                
            time.sleep(random.uniform(0.1, 0.4))
            
    except KeyboardInterrupt:
        print("\n🛑 Stopped testing stream.")
    finally:
        producer.flush()

if __name__ == "__main__":
    main()