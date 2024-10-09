"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

// Define the types
type Subject = "Math" | "Science" | "English";
type EducationLevel = "Kindergarten" | "Primary School" | "Secondary School";

const subjects: Subject[] = ["Math", "Science", "English"];
const topics: Record<Subject, string[]> = {
  Math: ["Algebra", "Geometry", "Calculus"],
  Science: ["Physics", "Chemistry", "Biology"],
  English: ["Grammar", "Literature", "Writing"],
};
const educationLevels: EducationLevel[] = [
  "Kindergarten",
  "Primary School",
  "Secondary School",
];

export default function StudentInterface() {
  const [question, setQuestion] = useState("");
  const [subject, setSubject] = useState<Subject>(subjects[0]);
  const [topic, setTopic] = useState(topics[subjects[0]][0]);
  const [educationLevel, setEducationLevel] = useState<EducationLevel>(
    educationLevels[0]
  );
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubjectChange = (value: string) => {
    const newSubject = value as Subject;
    if (topics[newSubject]) {
      setSubject(newSubject);
      setTopic(topics[newSubject][0]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResponse("");

    try {
      const res = await fetch("http://localhost:8000/chat/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          subject,
          topic,
          educationLevel,
          context: "This is a sample context",
        }),
      });

      if (!res.ok) {
        throw new Error("Failed to fetch the response from the server");
      }

      const data = await res.json();
      setResponse(data.response || "No response received.");
    } catch (error: any) {
      console.error("Error:", error);
      setError(
        error.message || "Sorry, there was an error processing your request."
      );
    }
    setLoading(false);
  };

  return (
    <Card className="w-full max-w-2xl mx-auto mt-10">
      <CardHeader>
        <CardTitle>LearnSphere AI Assistant</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <Input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your question..."
          />

          {/* Subject Select */}
          <Select value={subject} onValueChange={handleSubjectChange}>
            <SelectTrigger>{subject}</SelectTrigger>
            <SelectContent>
              {subjects.map((subject) => (
                <SelectItem key={subject} value={subject}>
                  {subject}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Topic Select */}
          <Select value={topic} onValueChange={(value) => setTopic(value)}>
            <SelectTrigger>{topic}</SelectTrigger>
            <SelectContent>
              {topics[subject].map((topic) => (
                <SelectItem key={topic} value={topic}>
                  {topic}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Education Level Select */}
          <Select
            value={educationLevel}
            onValueChange={(value) =>
              setEducationLevel(value as EducationLevel)
            }
          >
            <SelectTrigger>{educationLevel}</SelectTrigger>
            <SelectContent>
              {educationLevels.map((level) => (
                <SelectItem key={level} value={level}>
                  {level}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button type="submit" disabled={loading}>
            {loading ? (
              <>
                <span>Thinking...</span>
              </>
            ) : (
              "Submit"
            )}
          </Button>
        </form>

        {/* Display error if any */}
        {error && (
          <div className="mt-4 p-4 bg-red-100 rounded-md">
            <h3 className="font-semibold mb-2 text-red-600">Error:</h3>
            <p>{error}</p>
          </div>
        )}

        {/* Display the AI response */}
        {response && (
          <div className="mt-4 p-4 bg-gray-100 rounded-md">
            <h3 className="font-semibold mb-2">AI Response:</h3>
            <p>{response}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
