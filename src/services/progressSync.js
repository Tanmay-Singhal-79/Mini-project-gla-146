import { doc, setDoc, getDoc, collection, query, where, getDocs, updateDoc, serverTimestamp } from "firebase/firestore";
import { db, auth } from "../firebase";
import api from "./api";

/**
 * Syncs progress to BOTH the API and direct Firestore as a fallback.
 * This ensures "instant-load" reliability even if the local backend is down.
 */
export const syncProgress = async (stepTitle) => {
  const trimmedTitle = stepTitle.trim();
  const user = auth.currentUser;

  // 1. Always try the API first (for ML tracking etc.)
  try {
    const response = await api.progress.update(trimmedTitle);
    return response.data;
  } catch (apiError) {
    console.warn("API Sync failed, falling back to direct Firestore:", apiError);
    
    // 2. Direct Firestore Fallback
    if (!user) throw new Error("Authentication required for sync.");
    
    const safeTitle = trimmedTitle.replace(/[^a-zA-Z0-9]/g, "_");
    const progressId = `${user.uid}_${safeTitle}`;
    const progressRef = doc(db, "progress", progressId);
    
    const progressDoc = await getDoc(progressRef);
    const now = new Date();
    
    if (progressDoc.exists()) {
      if (progressDoc.data().status !== "completed") {
        await updateDoc(progressRef, {
          status: "completed",
          completed_at: serverTimestamp()
        });
      }
      return { ...progressDoc.data(), id: progressId, status: "completed" };
    } else {
      const newProgress = {
        user_id: user.uid,
        step_title: trimmedTitle,
        status: "completed",
        completed_at: serverTimestamp()
      };
      await setDoc(progressRef, newProgress);
      return { ...newProgress, id: progressId };
    }
  }
};

/**
 * Robust progress fetcher with API priority and Firestore fallback.
 */
export const getRobustProgress = async () => {
  try {
    const response = await api.progress.get();
    return response.data;
  } catch (apiError) {
    console.warn("API Get failed, falling back to direct Firestore:", apiError);
    const user = auth.currentUser;
    if (!user) return { completedItems: [], percentage: 0 };
    
    const q = query(
      collection(db, "progress"), 
      where("user_id", "==", user.uid), 
      where("status", "==", "completed")
    );
    
    const querySnapshot = await getDocs(q);
    const completedItems = [];
    querySnapshot.forEach((doc) => {
      const data = doc.data();
      completedItems.push({
        id: doc.id,
        step_title: data.step_title,
        status: data.status,
        completed_at: data.completed_at?.toDate() || new Date()
      });
    });
    
    // Default to 500 total steps for percentage
    const percentage = Math.round((completedItems.length / 500) * 1000) / 10;
    
    return { completedItems, percentage };
  }
};
