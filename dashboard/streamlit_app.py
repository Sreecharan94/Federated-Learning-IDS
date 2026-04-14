import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

from dashboard.local_metrics import LocalMetricsCollector

st.set_page_config(
    page_title="SOC Dashboard: FL-IDS Enterprise",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        --primary-color: #00d4ff;
        --background: #0a0e27;
        --card-bg: #1a1f3a;
        --text-primary: #e0e0e0;
        --high-severity: #ff006e;
        --med-severity: #ffbe0b;
        --low-severity: #8338ec;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: var(--text-primary);
    }
    
    .alert-high { color: var(--high-severity); font-weight: bold; }
    .alert-med { color: var(--med-severity); font-weight: bold; }
    .alert-low { color: var(--low-severity); font-weight: bold; }
    
    .metric-card {
        background: rgba(19, 30, 60, 0.4);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(0, 212, 255, 0.15);
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

class SOCDashboard:
    def __init__(self):
        self.collector = LocalMetricsCollector()
        self.collector.start_collection()
        time.sleep(1)
        
        self.severity_map = {
            'High': ['DDoS', 'PortScan', 'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP', 'DDoS attacks-LOIC-HTTP', 'DoS attacks-GoldenEye', 'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest', 'DoS attacks-Slowloris'],
            'Medium': ['Bot', 'Web attacks', 'Brute Force -XSS', 'Brute Force -Web'],
            'Low': ['Infiltration', 'FTP-BruteForce', 'SSH-Bruteforce']
        }
        if 'traffic_history' not in st.session_state:
            st.session_state.traffic_history = []
        if 'last_total' not in st.session_state:
            st.session_state.last_total = 0
            st.session_state.last_total = 0
            
    def get_attack_severity(self, attack_type):
        for severity, attacks in self.severity_map.items():
            if attack_type in attacks:
                return severity
        return 'Medium'
        
    def populate_alerts(self, live_alerts):
        alerts = []
        for a in live_alerts:
            sev = self.get_attack_severity(a['type'])
            alerts.append({'time': a['time'], 'type': a['type'], 'severity': sev, 'target': a['target']})
        return alerts

    def display_kpi_tier(self, sys_metrics, fed_metrics, dist):
        """Tier 1: Global Health & KPIs spanning full width"""
        final_acc = 0.0
        clients = 0
        if fed_metrics and len(fed_metrics) > 0:
            latest = fed_metrics[-1]
            final_acc = latest.get('accuracy', final_acc)
            clients = latest.get('active_clients', clients)
            
        total_threats = sum(count for attack, count in dist.items() if attack != 'Benign')
        total_traffic = sum(count for attack, count in dist.items())

        st.markdown("<h2 style='text-align: center; color: #00d4ff; margin-bottom: 30px;'>Federated Real-Time Intelligence Matrix</h2>", unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        acc_text = f"{final_acc:.2%}" if clients > 0 else "N/A (Training Offline)"
        
        c1.metric("Global FL Accuracy", acc_text)
        c2.metric("Total Threats Neutralized", f"{total_threats:,}")
        c3.metric("Total Processed Packets", f"{total_traffic:,}")
        c4.metric("Pre-Node System CPU", f"{sys_metrics.get('avg_cpu_percent', 0):.1f}%")

    def display_threat_intelligence(self, alerts, dist, fed_metrics):
        """Tier 2: Alerts panel (Left) and Attack Distribution Pie (Right)"""
        st.markdown("---")
        # Split half and half
        c_left, c_right = st.columns(2)
        
        with c_left:
            st.subheader("🚨 Priority Incident Feed")
            if not alerts:
                st.success("Network secure. Awaiting hostile anomalies.")
            else:
                for a in alerts[:6]:
                    color = "🔴" if a['severity'] == "High" else "🟡" if a['severity'] == "Medium" else "🔵"
                    class_name = f"alert-{'high' if a['severity'] == 'High' else 'med' if a['severity'] == 'Medium' else 'low'}"
                    st.markdown(f"**{color} [{a['time']}]** <span class='{class_name}'>{a['severity'].upper()} PERIMETER BREACH:</span> Hostile signature **{a['type']}** intercepted targeting **{a['target']}**", unsafe_allow_html=True)

        with c_right:
            st.subheader("🕸️ Global Attack Distribution")
            if dist:
                df = pd.DataFrame(list(dist.items()), columns=['Attack Type', 'Count'])
                fig_pie = px.pie(df, values='Count', names='Attack Type', hole=0.5)
                fig_pie.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0), height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Awaiting Kafka telemetry...")

    def display_timeline(self, total_traffic):
        """Tier 3: Network Traffic Timeline"""
        st.markdown("---")
        st.subheader("📡 Continuous Streaming Pipeline Monitor (Live)")
        
        # Calculate exactly how many packets were processed in the last 2 seconds
        current_inferences = total_traffic - st.session_state.last_total
        if current_inferences < 0: current_inferences = 0  # Handle reset
        
        st.session_state.last_total = total_traffic
        st.session_state.traffic_history.append({'time': datetime.now(), 'traffic': current_inferences})
        
        # Keep only the last 30 intervals (60 seconds of history)
        if len(st.session_state.traffic_history) > 30:
            st.session_state.traffic_history.pop(0)
            
        # Ensure we have at least something to map to avoid empty graphs
        plot_data = pd.DataFrame(st.session_state.traffic_history)
        if plot_data.empty:
            plot_data = pd.DataFrame([{'time': datetime.now(), 'traffic': 0}])
            
        fig_area = px.area(plot_data, x='time', y='traffic', 
                           title="Inferences per 2-second block",
                           color_discrete_sequence=['#00d4ff'])
        fig_area.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), height=250)
        st.plotly_chart(fig_area, use_container_width=True)
                
    def sidebar(self):
        with st.sidebar:
            st.header("🎯 Operations Console")
            
            if st.button("🔄 Manual Sync"):
                st.rerun()
            
            st.markdown("---")
            st.subheader("✅ Topology Hook")
            st.write("🟢 **Kafka Cluster:** 127.0.0.1:9092")
            st.write("🟢 **AI Pipeline:** LSTM Attention Bound")
            st.write("🟢 **Aggregator:** FedAvg Loop Locked")
            
            st.markdown("---")
            st.subheader("Privacy Guards")
            st.write("• **Transfer Mode:** Encrypted Weights")
            st.write("• **Raw Telemetry:** Locked On-Device")
            
            st.markdown("---")
            auto_refresh = st.checkbox("Live Auto-Refresh (2s)", value=True)
            return auto_refresh
            
    def run(self):
        auto_refresh = self.sidebar()
        
        dist = self.collector.get_overall_attack_distribution()
        raw_alerts = self.collector.get_latest_alerts()
        alerts = self.populate_alerts(raw_alerts)
        sys_metrics = self.collector.get_system_summary()
        fed_metrics = self.collector.get_recent_federated_metrics()
        
        # Build Tier 1-3 Layout smoothly
        total_traffic = sum(count for attack, count in dist.items())
        
        self.display_kpi_tier(sys_metrics, fed_metrics, dist)
        self.display_threat_intelligence(alerts, dist, fed_metrics)
        self.display_timeline(total_traffic)
        
        if auto_refresh:
            time.sleep(2)
            st.rerun()

if __name__ == "__main__":
    app = SOCDashboard()
    app.run()