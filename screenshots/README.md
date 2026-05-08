---

# 📷 Project Architecture & Results

## 🗄 Database Design
![Database Design](screenshots/Data%20base%20Design.jpg)

The database design represents the flow feature schema used for storing network traffic attributes, attack labels, and federated learning model inputs.

---

## 🔄 Live Streaming Pipeline
![Streaming Pipeline](screenshots/Live%20Streaming%20Pipe%20line.jpg)

The live streaming pipeline continuously processes real-time network traffic data using Apache Kafka and displays inference activity for intrusion detection operations.

---

## 🚨 Priority Incident Feed & Global Attack Distribution
![Attack Distribution](screenshots/Priority%20Incident%20Feed%20and%20global%20Attack%20Distribution.jpg)

This module displays live security incidents and attack classifications along with graphical attack distribution analysis for identifying malicious traffic behavior.

---

## 📊 Real-Time SOC Dashboard
![SOC Dashboard](screenshots/Real%20time%20soc%20Dashboard.jpg)

The Security Operations Center (SOC) dashboard provides real-time monitoring of network traffic, processed packets, system CPU usage, and threat neutralization statistics for intrusion detection analysis.

---

## 📈 Global FL Performance Metrics Dashboard
![FL Metrics](screenshots/global%20FL%20Performance%20Metrics%20Dashboard.jpg)

This dashboard visualizes the federated learning model performance including global FL accuracy, total processed packets, detected threats, and node-level system metrics.

---

## 🏗 System Architecture
![System Architecture](screenshots/system%20architecture.jpg)

This architecture illustrates the complete Federated Learning-based Intrusion Detection System (FL-IDS) integrating Kafka streaming, federated nodes, LSTM training, model aggregation, and real-time attack monitoring dashboard.

---

# 🧩 UML Diagrams

## UML Class Diagram
![Class Diagram](screenshots/UML%20Class%20Diagram.jpg)

The UML class diagram defines the structural relationships between FlowData, Client, KafkaStream, and Detector modules within the FL-IDS system.

---

## UML Sequence Diagram
![Sequence Diagram](screenshots/UML%20Sequence%20diagram.jpg)

The sequence diagram illustrates the communication flow between client nodes, Kafka streaming pipeline, federated server, detector module, and monitoring dashboard.

---

## UML Use Case Diagram
![Use Case](screenshots/UML%20Use%20case%20.jpg)

The UML use case diagram represents interactions between the federated server, client nodes, and security analyst for intrusion detection workflow management.
