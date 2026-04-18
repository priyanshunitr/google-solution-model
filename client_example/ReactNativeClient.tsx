/**
 * SOS Detection Service - React Native client example.
 *
 * Shows the full upload -> poll -> download flow.
 *
 * Dependencies:
 *   npm install expo-image-picker expo-file-system
 *
 * Usage:
 *   Replace API_BASE with your deployed server URL. For Android emulator
 *   calling a local server, use http://10.0.2.2:8000 (not localhost).
 */

import React, { useState } from "react";
import {
  View,
  Text,
  Button,
  StyleSheet,
  ActivityIndicator,
  Alert,
  ScrollView,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import * as FileSystem from "expo-file-system";

const API_BASE = "http://YOUR_SERVER_IP:8000"; // <-- CHANGE ME

type JobStatus = {
  job_id: string;
  status: "queued" | "running" | "done" | "failed";
  progress: number;
  message?: string;
  n_alerts: number;
  fps_processed?: number;
  video_url?: string;
  alerts_url?: string;
  error?: string;
};

export default function SOSClient() {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [log, setLog] = useState<string[]>([]);

  const appendLog = (line: string) =>
    setLog((prev) => [...prev.slice(-20), line]);

  const pickAndUpload = async () => {
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) {
      Alert.alert("Permission required", "Allow media library access.");
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      allowsEditing: false,
      quality: 1,
    });
    if (result.canceled) return;

    const asset = result.assets[0];
    setBusy(true);
    appendLog(`Selected: ${asset.fileName ?? asset.uri}`);

    try {
      // ---- Upload ----
      const form = new FormData();
      form.append("video", {
        uri: asset.uri,
        name: asset.fileName ?? "upload.mp4",
        type: asset.mimeType ?? "video/mp4",
      } as any);
      // Optional: pass calibration
      // form.append("calibration", JSON.stringify({
      //   src_points: [[520,380],[760,380],[1100,680],[180,680]],
      //   real_width_m: 7.0,
      //   real_height_m: 25.0,
      // }));

      appendLog("Uploading...");
      const uploadRes = await fetch(`${API_BASE}/jobs`, {
        method: "POST",
        body: form,
      });
      if (!uploadRes.ok) {
        const text = await uploadRes.text();
        throw new Error(`Upload failed ${uploadRes.status}: ${text}`);
      }
      const { job_id } = await uploadRes.json();
      appendLog(`Job queued: ${job_id}`);

      // ---- Poll ----
      let st: JobStatus | null = null;
      while (true) {
        await new Promise((r) => setTimeout(r, 2000));
        const r = await fetch(`${API_BASE}/jobs/${job_id}`);
        st = (await r.json()) as JobStatus;
        setStatus(st);
        appendLog(
          `[${st.status}] ${(st.progress * 100).toFixed(0)}% - ${st.message ?? ""}`
        );
        if (st.status === "done" || st.status === "failed") break;
      }

      if (st!.status === "failed") {
        Alert.alert("Job failed", st!.error ?? "unknown");
        return;
      }

      // ---- Download outputs ----
      const videoDest = `${FileSystem.documentDirectory}sos_${job_id}.mp4`;
      const alertsDest = `${FileSystem.documentDirectory}sos_${job_id}.json`;

      appendLog("Downloading video...");
      await FileSystem.downloadAsync(`${API_BASE}${st!.video_url}`, videoDest);
      appendLog("Downloading alerts...");
      await FileSystem.downloadAsync(`${API_BASE}${st!.alerts_url}`, alertsDest);

      appendLog(`Saved video: ${videoDest}`);
      appendLog(`Saved alerts: ${alertsDest}`);
      appendLog(`Total alerts: ${st!.n_alerts}`);
      Alert.alert("Done", `${st!.n_alerts} alerts detected.`);
    } catch (e: any) {
      Alert.alert("Error", e.message ?? String(e));
      appendLog(`ERROR: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>SOS Detection</Text>
      <Button
        title={busy ? "Processing..." : "Pick video & upload"}
        onPress={pickAndUpload}
        disabled={busy}
      />
      {busy && <ActivityIndicator size="large" style={{ marginTop: 12 }} />}
      {status && (
        <View style={styles.statusBox}>
          <Text>Status: {status.status}</Text>
          <Text>Progress: {(status.progress * 100).toFixed(0)}%</Text>
          {status.fps_processed != null && (
            <Text>FPS: {status.fps_processed.toFixed(1)}</Text>
          )}
          <Text>Alerts: {status.n_alerts}</Text>
        </View>
      )}
      <View style={styles.logBox}>
        {log.map((l, i) => (
          <Text key={i} style={styles.logLine}>
            {l}
          </Text>
        ))}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 24, paddingTop: 60 },
  title: { fontSize: 22, fontWeight: "700", marginBottom: 16 },
  statusBox: {
    marginTop: 20,
    padding: 12,
    backgroundColor: "#eef",
    borderRadius: 8,
  },
  logBox: {
    marginTop: 20,
    padding: 12,
    backgroundColor: "#111",
    borderRadius: 8,
    minHeight: 200,
  },
  logLine: { color: "#0f0", fontFamily: "Courier", fontSize: 11 },
});
