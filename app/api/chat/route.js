import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = 
`
You are a RateMyProfessor assistant designed to help students find the best professors based on their specific needs. When a user asks for recommendations or queries about professors, you will search through a database of professor reviews using Retrieval-Augmented Generation (RAG). For each query, retrieve and analyze the top 3 professors who best match the student's criteria. Provide a concise summary of each professor, including their name, subject, overall rating, and a brief excerpt from their reviews that highlights their teaching style, strengths, and any relevant details that align with the user's query.

Ensure your responses are clear, helpful, and tailored to the student's query.
If the student provides specific preferences (e.g., difficulty level, teaching style), prioritize professors that best fit those criteria.
If no specific criteria are provided, select the top-rated professors in the relevant subject area.
Always provide the top 3 professors based on the most relevant and up-to-date information.
Include a brief conclusion summarizing why these professors were chosen.
`

export async function POST(req){
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY, 
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length-1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small', 
        input: text, 
        encoding_format: 'float', 
    })

    const results = await index.query({
        topK: 5, 
        includeMetadata: true, 
        vector: embedding.data[0].embedding,
    })

    let resultString = ''
    results.matches.forEach((match) => {
        resultString += `
        Returned Results:
        ProfessorL ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        /n/n`
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length -1)
    const completion = await openai.chat.completions.create({
        messages: [
            {role: 'system', content: systemPrompt}, 
            ...lastDataWithoutLastMessage, 
            {role:'user', content: lastMessageContent}, 
        ], 
        model: 'gpt-4o-mini', 
        stream: true,
    })

    const stream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder()
          try {
            for await (const chunk of completion) {
              const content = chunk.choices[0]?.delta?.content
              if (content) {
                const text = encoder.encode(content)
                controller.enqueue(text)
              }
            }
          } catch (err) {
            controller.error(err)
          } finally {
            controller.close()
          }
        },
      })
      
      return new NextResponse(stream)
}