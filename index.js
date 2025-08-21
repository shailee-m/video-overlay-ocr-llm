// Dependencies:
// npm install fluent-ffmpeg fs-extra node-wav webrtcvad @google-cloud/text-to-speech node-fetch form-data openai googleapis
// https://www.youtube.com/watch?v=32HbEn4qzJ0&pp=ygUybGVjdHVyZSBvbiBmYWN0b3JzIGFuZCBtdWx0aXBsZXMgImd1amFyYXRpIiBtZWRpdW0%3D
//  https://www.youtube.com/watch?v=MfCZxNrY2b0

const fs = require("fs-extra");
const path = require("path");
const ffmpeg = require("fluent-ffmpeg");
const wav = require("node-wav");
const Vad = require("webrtcvad");
const textToSpeech = require("@google-cloud/text-to-speech");
// const fetch = require("node-fetch").fetch;
const fetch = (...args) =>
  import("node-fetch").then(({ default: fetch }) => fetch(...args));
const FormData = require("form-data");
const { OpenAI } = require("openai");
const { google } = require("googleapis");
const GoogleGenerativeAI = require("@google/generative-ai").GoogleGenerativeAI;

const client = new textToSpeech.TextToSpeechClient();
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const SarvamAIClient = require("sarvamai").SarvamAIClient;

const inputVideo = "input.mp4";
const tempDir = "./temp";
const audioWav = path.join(tempDir, "audio.wav");
const outputDir = "./output";
const outputAudio = path.join(outputDir, "final_audio.wav");

async function extractAudio() {
  await fs.ensureDir(tempDir);
  return new Promise((resolve, reject) => {
    ffmpeg(inputVideo)
      .noVideo()
      .audioCodec("pcm_s16le")
      .audioChannels(1)
      .audioFrequency(16000)
      .save(audioWav)
      .on("end", resolve)
      .on("error", reject);
  });
}

async function transcribeWithWhisper(audioPath) {
  const form = new FormData();
  form.append("file", fs.createReadStream(audioPath));
  form.append("model", "whisper-1");
  form.append("response_format", "verbose_json");

  const response = await fetch(
    "https://api.openai.com/v1/audio/transcriptions",
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: form,
    }
  );

  if (!response.ok) {
    throw new Error(`Whisper API error: ${response.statusText}`);
  }

  const result = await response.json();
  return result.segments.map((s, i) => ({
    id: i,
    text: s.text.trim(),
    start: s.start,
    end: s.end,
  }));
}

async function translateWithSarvam(text) {
  const client = new SarvamAIClient({
    apiSubscriptionKey: "2d161a3d-533a-44e2-a10a-17d75071e727",
  });
  const response = await client.text.translate({
    input: text,
    source_language_code: "en-IN",
    target_language_code: "gu-IN",
    speaker_gender: "Male",
  });

  // const response = await fetch("https://api.sarvam.ai/v1/translate", {
  //   method: "POST",
  //   headers: {
  //     // 'Authorization': `Bearer ${process.env.SARVAM_API_KEY}`,
  //     "api-subscription-key": `2d161a3d-533a-44e2-a10a-17d75071e727`,
  //     "Content-Type": "application/json",
  //   },
  //   body: JSON.stringify({
  //     source_language_code: "en",
  //     target_language_code: "gu",
  //     input: text,
  //     speaker_gender: "Male",
  //   }),
  // });

  // if (!response.ok) {
  //   throw new Error(`Sarvam API error: ${response.statusText}`);
  // }

  // const result = await response.json();
  if (!response?.translated_text) {
    throw new Error("Sarvam API did not return translated text.");
  }
  return response?.translated_text || "";
}

async function translateTranscriptWithSarvam(segments) {
  const translations = [];

  for (const seg of segments) {
    const translatedText = await translateWithSarvam(seg.text);

    // Estimate speed based on segment duration (optional refinement)
    const duration = seg.end - seg.start;
    const wordsPerSecond = translatedText.split(/\s+/).length / duration;
    const targetSpeed = Math.min(Math.max(wordsPerSecond / 2.17, 0.25), 2);

    translations.push({
      text: translatedText,
      speed: targetSpeed,
    });
  }

  return translations;
}

async function generateWithSarvam(prompt, context) {
  // return [
  //   {
  //     text: "18 ટકા ને દશાંશ અને સરળતમ ભિન્નરાશિમાં લખવાનું છે.",
  //     speed: 1.0,
  //   },
  //   {
  //     text: "તો પહેલા આપણે દશાંશ સ્વરૂપમાં લખીએ.",
  //     speed: 1.2,
  //   },
  //   {
  //     text: "18 ટકા એટલે 18 પ્રતિ 100, અથવા 18 પ્રતિ સો.",
  //     speed: 1.0,
  //   },
  //   {
  //     text: "અહીં 'પ્રતિ' શબ્દને અલગ કરવામાં આવ્યો છે.",
  //     speed: 1.5,
  //   },
  //   {
  //     text: "જોકે તે એક જ શબ્દ છે, પણ હું તેને 'પ્રતિ' તરીકે લખી રહ્યો છું.",
  //     speed: 1.2,
  //   },
  //   {
  //     text: "'સેન્ટ' નો અર્થ પણ 100 થાય છે.",
  //     speed: 1.0,
  //   },
  //   {
  //     text: "તેથી આનો અર્થ 18 પ્રતિ 100 થાય છે.",
  //     speed: 1.0,
  //   },
  //   {
  //     text: "મેં પહેલા દશાંશ સ્વરૂપમાં કરવાનું કહ્યું હતું, પણ આપણે સીધું ભિન્નરાશિ સ્વરૂપમાં લખી શકીએ છીએ.",
  //     speed: 1.0,
  //   },
  //   {
  //     text: "18 પ્રતિ 100 એટલે ભિન્નરાશિમાં 18/100.",
  //     speed: 1.2,
  //   },
  //   {
  //     text: "હવે આપણે દશાંશ સ્વરૂપમાં કરીએ કે સીધું ભિન્નરાશિ સ્વરૂપમાં કરીએ.",
  //     speed: 1.0,
  //   },
  //   {
  //     text: "ભિન્નરાશિ સ્વરૂપમાં સરળ બનાવવા માટે, આપણે જોઈશું કે 18 અને 100 નો સામાન્ય અવયવ છે કે નહીં.",
  //     speed: 1.0,
  //   },
  //   {
  //     text: "બંને સંખ્યાઓ સમ છે, તેથી 2 વડે ભાગી શકાય છે.",
  //     speed: 1.2,
  //   },
  //   {
  //     text: "તેથી અંશ અને છેદ બંનેને 2 વડે ભાગીએ.",
  //     speed: 1.0,
  //   },
  //   {
  //     text: "18 ભાગ્યા 2 એટલે 9, અને 100 ભાગ્યા 2 એટલે 50.",
  //     speed: 1.2,
  //   },
  //   {
  //     text: "9 અને 50 નો કોઈ સામાન્ય અવયવ નથી.",
  //     speed: 1.0,
  //   },
  //   {
  //     text: "તેથી 9/50 એ સરળતમ ભિન્નરાશિ છે.",
  //     speed: 1.2,
  //   },
  //   {
  //     text: "દશાંશ સ્વરૂપમાં, 18% એ 0.18 થાય છે.",
  //     speed: 1.0,
  //   },
  //   {
  //     text: "આને 1 દશાંશ અને 8 સોત્રીસ તરીકે પણ લખી શકાય.",
  //     speed: 1.2,
  //   },
  //   {
  //     text: "આ સરળતમ ભિન્નરાશિ 9/50 જેટલું જ છે.",
  //     speed: 1.0,
  //   },
  // ];
  // Initialize the SarvamAI client with your API key
  const client = new SarvamAIClient({
    apiSubscriptionKey: "2d161a3d-533a-44e2-a10a-17d75071e727",
  });

  const response = await client.chat.completions({
    max_tokens: 8192,
    // model: "sarvamai-llama-3.1-70b",
    messages: [
      {
        role: "system",
        content: "You are a translation assistant that outputs only JSON.",
      },
      { role: "user", content: `${prompt}\n\n${context}` },
    ],
  });
  if (!response || !response.choices || response.choices.length === 0) {
    throw new Error("Sarvam AI did not return a valid response.");
  }
  let responseText = "";
  try {
    responseText =
      JSON.parse(response.choices[0]?.message?.content.replaceAll("\n", "")) ||
      "";
  } catch (e) {
    console.error("Failed to parse JSON from Sarvam response:", e);
    throw new Error("Failed to parse JSON from Sarvam response.");
  }

  return responseText;
}

async function callLLMTranslationModel({ model = "openai", prompt, context }) {
  if (model === "openai") {
    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "system",
          content: "You are a translation assistant that outputs only JSON.",
        },
        { role: "user", content: `${prompt}\n\n${context}` },
      ],
      temperature: 0.4,
    });
    const jsonText = response.choices[0].message.content;
    try {
      return JSON.parse(jsonText);
    } catch (err) {
      throw new Error("Failed to parse JSON from OpenAI response.");
    }
  } else if (model === "gemini") {
    const gemini_api_key = process.env.API_KEY;
    const googleAI = new GoogleGenerativeAI(gemini_api_key);
    const geminiConfig = {
      temperature: 0.9,
      topP: 1,
      topK: 1,
      maxOutputTokens: 4096,
    };

    const geminiModel = googleAI.getGenerativeModel({
      model: "gemini-2.5-flash",
      geminiConfig,
    });

    const result = await geminiModel.generateContent(`${prompt}\n\n${context}`);
    const response = result.response;
    console.log(response.text());

    const text = response.text() || "{}";
    try {
      return JSON.parse(text.replace("```json", "").replace("```", ""));
    } catch (err) {
      throw new Error("Failed to parse JSON from Gemini response.");
    }
  } else if (model === "sarvam") {
    return await generateWithSarvam(prompt, context);
  } else {
    throw new Error("Unsupported model specified.");
  }
}

async function synthesizeSpeech(text, speed, index) {
  const request = {
    input: { text },
    voice: {
      languageCode: "gu-IN",
      ssmlGender: 1 /*, name: "gu-IN-chirp3-HD-Archid" */,
    },
    audioConfig: { audioEncoding: "MP3", speakingRate: speed },
  };
  const [response] = await client.synthesizeSpeech(request);
  const filePath = path.join(outputDir, `segment_${index}.mp3`);
  await fs.writeFile(filePath, response.audioContent, "binary");
  return filePath;
}

async function generateSilence(duration, index) {
  return new Promise((resolve, reject) => {
    const filePath = path.join(outputDir, `silence_${index}.mp3`);
    ffmpeg()
      .input("anullsrc")
      .inputFormat("lavfi")
      .duration(duration)
      .output(filePath)
      .on("end", () => resolve(filePath))
      .on("error", reject)
      .run();
  });
}

function calculateGoogleTtsSpeed(startSeconds, endSeconds, text) {
  const duration = endSeconds - startSeconds;
  if (duration <= 0) throw new Error("Duration must be greater than 0");

  const wordCount = text.trim().split(/\s+/).length;
  const targetWPM = (wordCount / duration) * 60;
  const defaultWPM = 180;

  let speed = targetWPM / defaultWPM;

  // Clamp between 0.25 and 2.0
  speed = Math.max(0.25, Math.min(speed, 2.0));

  // Round to 2 decimal places
  return parseFloat(speed.toFixed(2));
}
/* 
function getMaxTargetWords(sourceText, start, end, adjustmentFactor = 1.1) {
  const duration = end - start;
  const sourceWords = sourceText.trim().split(/\s+/).length;
  const sourceSpeed = sourceWords / duration;

  const maxWords = Math.floor(sourceSpeed * duration * adjustmentFactor);
  // return Math.max(1, maxWords); // Ensure at least 1 word
  return { MaxWords: maxWords, sourceSpeed }; // Ensure at least 1 word
} */

function getMaxTargetWords(sourceText, start, end, adjustmentFactor = 1.1) {
  const duration = end - start;
  const sourceWords = sourceText.trim().split(/\s+/).length;
  const sourceSpeed = sourceWords / duration;

  // Google TTS range mapping
  const minWPS = 1,
    maxWPS = 4;
  const minRate = 0.5,
    maxRate = 1.5;
  let ttsRate =
    ((sourceSpeed - minWPS) / (maxWPS - minWPS)) * (maxRate - minRate) +
    minRate;
  ttsRate = Math.max(minRate, Math.min(maxRate, ttsRate)); // Clamp between 0.5 and 2

  const maxWords = Math.floor(sourceSpeed * duration * adjustmentFactor);
  return { MaxWords: maxWords, sourceSpeed, ttsRate };
}

async function processTranscriptWithWhisperAlignment(segments, translations) {
  // const audioParts = [];
  // for (let i = 0; i < segments.length; i++) {
  //   const seg = segments[i];
  //   const duration = seg.end - seg.start;
  //   const sentence = translations[i]?.text || "";
  //   const speed = translations[i]?.speed || 1.0;

  //   if (!sentence.trim()) {
  //     const silenceFile = await generateSilence(duration, `silence_${i}`);
  //     audioParts.push(silenceFile);
  //   } else {
  //     const voiceFile = await synthesizeSpeech(sentence, speed, `voice_${i}`);
  //     audioParts.push(voiceFile);
  //   }
  // }
  // return audioParts;
  const audioParts = [];
  let lastSpokenIndex = -1;

  for (let i = 0; i < segments.length; i++) {
    const seg = segments[i];
    const duration = seg.end - seg.start;
    const sentence = translations[i]?.text || "";
    // const calculatedSpeed = calculateGoogleTtsSpeed(
    //   seg.start,
    //   seg.end,
    //   translations[i]?.text || ""
    // );
    const speed = segments[i]?.ttsRate;
    if (sentence.trim()) {
      lastSpokenIndex = i;
    }

    if (!sentence.trim()) {
      const silenceFile = await generateSilence(duration, `silence_${i}`);
      audioParts.push({ file: silenceFile, index: i });
    } else {
      const voiceFile = await synthesizeSpeech(sentence, speed, `voice_${i}`);
      audioParts.push({ file: voiceFile, index: i });

      // Check duration of the generated voice file
      const ttsDuration = await new Promise((resolve, reject) => {
        ffmpeg.ffprobe(voiceFile, (err, metadata) => {
          if (err) reject(err);
          else resolve(metadata.format.duration);
        });
      });

      const remaining = duration - ttsDuration;

      if (remaining > 0.25) {
        // Add trailing silence to fill the gap
        const silenceFile = await generateSilence(remaining, `pad_${i}`);

        audioParts.push({ file: silenceFile, index: i + 0.5 });
      }
    }
  }

  // const filteredParts = audioParts.filter((p) => p.index <= lastSpokenIndex);
  return audioParts.map((p) => p.file);
}

async function mergeAudioParts(parts) {
  return new Promise((resolve, reject) => {
    const ff = ffmpeg();
    parts.forEach((p) => ff.input(p));
    ff.mergeToFile(outputAudio, tempDir)
      .on("end", () => {
        console.log("✅ Final audio saved:", outputAudio);
        resolve();
      })
      .on("error", reject);
  });
}

async function muteVideoAndAddFinalAudio(
  inputVideoPath,
  finalAudioPath,
  outputVideoPath
) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputVideoPath)
      .outputOptions("-an") // Remove original audio
      .output("temp_muted.mp4")
      .on("end", () => {
        ffmpeg()
          .input("temp_muted.mp4")
          .input(finalAudioPath)
          .outputOptions([
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-shortest",
          ])
          .output(outputVideoPath)
          .on("end", () => {
            console.log(
              "✅ Final video with new audio saved:",
              outputVideoPath
            );
            fs.removeSync("temp_muted.mp4");
            resolve();
          })
          .on("error", reject)
          .run();
      })
      .on("error", reject)
      .run();
  });
}

// Rough algorithm (pseudo-JS):
function groupSegmentsIntoSentences(segments) {
  let sentence = "";
  let start = segments[0].start;
  let grouped = [];

  for (let i = 0; i < segments.length; i++) {
    sentence += segments[i].text + " ";

    if (
      sentence.trim().endsWith(".") ||
      sentence.trim().endsWith("?") ||
      sentence.trim().endsWith("!")
    ) {
      grouped.push({
        text: sentence.trim(),
        start: start,
        end: segments[i].end,
      });
      if (i + 1 < segments.length) {
        start = segments[i + 1].start;
      }
      sentence = "";
    }
  }

  return grouped;
}

async function main() {
  await fs.ensureDir(outputDir);
  await extractAudio();
  const segments = groupSegmentsIntoSentences(
    await transcribeWithWhisper(audioWav)
  ).map((s, i) => ({
    id: i,
    text: s.text.trim(),
    start: s.start,
    end: s.end,
    ...getMaxTargetWords(s.text, s.start, s.end),
  }));
  const videoMinutes = (segments.at(-1).end / 60).toFixed(1);
  const sarvamPrompt =
    `YOu are a translator for teaching videos. In the context is the full transcript of a video explained in english and it is ${videoMinutes} minutes long.\n\n` +
    `The transcript is divided into segments, each with a start and end time. The start and end time depict time taken to say the sentence. Each segment has a key "MaxWords" which indicates the maximum number of words that can be used in the Gujarati translation for that segment.\n\n` +
    `Your task is generate gujarati translation so that I can put gujarati voice over on the same video.\n\n` +
    `1. At any point, the Gujarati transcript must explain exactly what was being in explained in the original video\n` +
    `2. Gujarati transcript must be as long as the video. In the given context, each sentence has a key: "MaxWords". The translated section should STRICTLY not exceed this length\n` +
    `3. A word-by-word translation of English to Gujarati may not always make sense. Understand the context and make a meaningful Gujarati sentence that means the same. it doesn't have to be exact translation\n` +
    `\n` +
    `4. Always give json response` +
    `Expected output format:\n` +
    `[\n` +
    `  {\n` +
    `    "text": "Gujarati sentence here",\n` +
    `  },\n` +
    `  // ... more segments\n` +
    `]\n ` +
    `Context:\n`;

  const geminiPrompt =
    `YOu are a translator for teaching videos. In the context is the full transcript of a video explained in english and it is ${videoMinutes} minutes long.\n\n` +
    `The transcript is divided into segments, each with a start and end time. The start and end time depict time taken to say the sentence. Each segment has a key "MaxWords" which indicates the maximum number of words that can be used in the Gujarati translation for that segment.\n\n` +
    `Your task is generate gujarati translation so that I can put gujarati voice over on the same video.\n\n` +
    `1. At any point, the Gujarati transcript must explain exactly what was being in explained in the original video \n` +
    `2. Gujarati transcript must be create so that the audio as long as that of the original. In the given context, each sentence has a key: "MaxWords". The translated sentence should STRICTLY NOT exceed these maxwords \n` +
    `3. A word-by-word translation of English to Gujarati may not always make sense. Understand the context and make a meaningful Gujarati sentence that means the same. A word by word translation is not necessary as long as the meaning of the sentence is maintained.\n` +
    `\n` +
    `4. Feel free to omit irrelevant text if the text to be spoken may exceed  ${videoMinutes} minutes.  \n` +
    `5. Always give json response. Output array should be eqaul to the length  of the givencontext array, which is ${segments.length} in size. \n` +
    `Expected output format:\n` +
    `[\n` +
    `  {\n` +
    `    "text": "Gujarati sentence here",\n` +
    `  },\n` +
    `  // ... more segments\n` +
    `]\n ` +
    `Context:\n`;

  // const context = segments.map((s, i) => `${i + 1}. ${s.text}`).join(" ");
  const gujaratiTranslations = await callLLMTranslationModel({
    model: "gemini", // "sarvam", // or "openai" or "gemini"
    prompt: geminiPrompt, // or sarvamPrompt
    context: JSON.stringify(segments),
  });
  console.log("Translations:", gujaratiTranslations);
  // const gujaratiTranslations = await translateTranscriptWithSarvam(segments);
  const audioParts = await processTranscriptWithWhisperAlignment(
    segments,
    gujaratiTranslations
  );
  await mergeAudioParts(audioParts);
  muteVideoAndAddFinalAudio(
    inputVideo,
    outputAudio,
    path.join(outputDir, "final_video.mp4")
  ).catch(console.error);
}

main().catch(console.error);

/*
Example expected JSON from LLM:
[
  {
    "text": "18 ટકા ને ડેસિમલ અને ફ્રેક્શન રૂપમાં કેવી રીતે લખવું તે સમજાવાયું છે.",
    "speed": 0.92
  },
  {
    "text": "ચાલો તેને ડેસિમલ તરીકે બદલીયે.",
    "speed": 0.84
  },
  {
    "text": "18 ટકા એટલે 100 માંથી 18 ભાગ થાય.",
    "speed": 1.05
  }
  // ... continued for all segments
]
*/
