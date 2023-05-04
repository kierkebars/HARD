/*
  ==============================================================================

    This file contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
HARDAudioProcessor::HARDAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                       .withInput ("Sidechain", juce::AudioChannelSet::stereo(), true)
                     #endif
                       )
#endif
,parameters(*this, nullptr, juce::Identifier("HARDPlugin"),
            {
    std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("harmony", 1),
                                                "Harmony",
                                                0.0f, 1.0f, 0.0f),
    std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("rhythm", 1),
                                                "Rhythm",
                                                0.0f, 1.0f, 0.0f),
    std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("sourceGain", 1),
                                                "Source Gain",
                                                0.0f, 1.0f, 1.0f),
    std::make_unique<juce::AudioParameterFloat>(juce::ParameterID("sidechainGain", 1),
                                                "Sidechain Gain",
                                                0.0f, 1.0f, 1.0f),
    std::make_unique<juce::AudioParameterBool>(juce::ParameterID("sync",1),
                                               "Link Sliders",
                                               false)
})
{
    fifoBufferIn1.clearBuffer();
    fifoBufferIn2.clearBuffer();
    fifoBufferOutDNN.clearBuffer();
    setLatencySamples(OUTPUT_DELAY_SAMPLES-OUTPUT_DELAY_BIAS_SAMPLES);
    pInferenceThread = new ONNXMorpherInferenceThread();

    harmonyParameter = parameters.getRawParameterValue("harmony");
    rhythmParameter = parameters.getRawParameterValue("rhythm");
    sourceGainParameter = parameters.getRawParameterValue("sourceGain");
    sidechainGainParameter = parameters.getRawParameterValue("sidechainGain");
    syncParameter = parameters.getRawParameterValue("sync");
}

HARDAudioProcessor::~HARDAudioProcessor()
{
}

//==============================================================================
const juce::String HARDAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool HARDAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool HARDAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool HARDAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double HARDAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int HARDAudioProcessor::getNumPrograms()
{
    return 1;
}

int HARDAudioProcessor::getCurrentProgram()
{
    return 0;
}

void HARDAudioProcessor::setCurrentProgram (int index)
{
}

const juce::String HARDAudioProcessor::getProgramName (int index)
{
    return {};
}

void HARDAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
}

//==============================================================================
void HARDAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    fifoBufferIn1.clearBuffer();
    fifoBufferIn2.clearBuffer();
    fifoBufferOutDNN.clearBuffer();
    fifoBufferOutDNN.fillZeros(OUTPUT_DELAY_SAMPLES);
}

void HARDAudioProcessor::releaseResources()
{
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool HARDAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void HARDAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    
    auto numSamples = buffer.getNumSamples();
    
    auto mainInputOutput = getBusBuffer(buffer, true, 0);
    auto sideChainInput = getBusBuffer(buffer, true, 1);
    
    bool isSyncMode = *syncParameter > 0.5f;

    {
        const juce::ScopedLock lock (critical);
        fifoBufferIn1.pushData(mainInputOutput.getWritePointer(0), mainInputOutput.getWritePointer(1),  numSamples);
        fifoBufferIn2.pushData(sideChainInput.getWritePointer(0), sideChainInput.getWritePointer(1),  numSamples);
        numNewInputSamples += numSamples;
    }
    
    if (isSyncMode)
    {
        if(*harmonyParameter != preHarmonyParam)
        {
            *rhythmParameter = (float)*harmonyParameter;
        }
        else if(*rhythmParameter != preRhythmParam)
        {
            *harmonyParameter = (float)*rhythmParameter;
        }
    }
    
    
    if ((numNewInputSamples >= DNN_INPUT_SAMPLES) and (fifoBufferIn1.getBufferSize() >= (DNN_INPUT_SAMPLES+DNN_INPUT_CACHE_SAMPLES)) and (!pInferenceThread->threadIsInferring()))
    {
        const juce::ScopedLock lock (critical);
        jassert(fifoBufferIn1.getBufferSize() >= (DNN_INPUT_SAMPLES + DNN_INPUT_CACHE_SAMPLES));
        fifoBufferIn1.readData(dnnInputData1.data(), DNN_INPUT_SAMPLES + DNN_INPUT_CACHE_SAMPLES, DNN_INPUT_SAMPLES);
        fifoBufferIn2.readData(dnnInputData2.data(), DNN_INPUT_SAMPLES + DNN_INPUT_CACHE_SAMPLES, DNN_INPUT_SAMPLES);
        
        pInferenceThread->requestInference(dnnInputData1.data(), dnnInputData2.data(), *rhythmParameter, *harmonyParameter, *syncParameter);
        numNewInputSamples -= DNN_INPUT_SAMPLES;
    }

    {
        const juce::ScopedLock lock (critical);
        if (fifoBufferOutDNN.getBufferSize() >= numSamples)
        {
            fifoBufferOutDNN.readData(buffer, 0, numSamples);
        }
        else
        {
            buffer.clear();
        }
    }

    preRhythmParam = *rhythmParameter;
    preHarmonyParam = *harmonyParameter;
}

//==============================================================================
bool HARDAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* HARDAudioProcessor::createEditor()
{
    return new HARDAudioProcessorEditor (*this);
}

//==============================================================================
void HARDAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void HARDAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));

    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName (parameters.state.getType()))
            parameters.replaceState (juce::ValueTree::fromXml (*xmlState));
}


