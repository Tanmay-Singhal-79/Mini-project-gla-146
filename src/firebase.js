import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyC6XZlB2cX4v83B3gDiSc1wzUbOpxriEIE",
  authDomain: "learnpath-ai-27d0e.firebaseapp.com",
  projectId: "learnpath-ai-27d0e",
  storageBucket: "learnpath-ai-27d0e.firebasestorage.app",
  messagingSenderId: "760413957301",
  appId: "1:760413957301:web:f0ebd36bf878d4845a22fc"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);
