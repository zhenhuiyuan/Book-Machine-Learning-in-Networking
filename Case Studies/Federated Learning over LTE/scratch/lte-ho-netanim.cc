#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netanim-module.h"
#include "ns3/network-module.h"
#include "ns3/node-list.h"
#include "ns3/point-to-point-module.h"

#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("LteHoNetanim");

// ---------- Trace callbacks ----------
static void
NotifyConnectionEstablishedUe(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    std::cout << Simulator::Now().GetSeconds() << "s " << context
              << " UE IMSI=" << imsi << " CellId=" << cellId << " RNTI=" << rnti
              << " connected" << std::endl;
}

static void
NotifyConnectionEstablishedEnb(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    std::cout << Simulator::Now().GetSeconds() << "s " << context
              << " eNB CellId=" << cellId << " UE IMSI=" << imsi << " RNTI=" << rnti
              << " connected" << std::endl;
}

static void
NotifyHandoverStartUe(std::string context,
                      uint64_t imsi,
                      uint16_t cellId,
                      uint16_t rnti,
                      uint16_t targetCellId)
{
    std::cout << Simulator::Now().GetSeconds() << "s " << context
              << " UE IMSI=" << imsi << " RNTI=" << rnti
              << " handover start: " << cellId << " -> " << targetCellId << std::endl;
}

static void
NotifyHandoverStartEnb(std::string context,
                       uint64_t imsi,
                       uint16_t cellId,
                       uint16_t rnti,
                       uint16_t targetCellId)
{
    std::cout << Simulator::Now().GetSeconds() << "s " << context
              << " eNB CellId=" << cellId << " UE IMSI=" << imsi << " RNTI=" << rnti
              << " handover start to CellId=" << targetCellId << std::endl;
}

static void
NotifyHandoverEndOkUe(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    std::cout << Simulator::Now().GetSeconds() << "s " << context
              << " UE IMSI=" << imsi << " RNTI=" << rnti
              << " handover completed at CellId=" << cellId << std::endl;
}

static void
NotifyHandoverEndOkEnb(std::string context, uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    std::cout << Simulator::Now().GetSeconds() << "s " << context
              << " eNB CellId=" << cellId << " completed handover of UE IMSI=" << imsi
              << " RNTI=" << rnti << std::endl;
}

int
main(int argc, char* argv[])
{
    // ---- Scenario parameters ----
    uint16_t numberOfUes = 1;
    uint16_t numberOfEnbs = 2;
    uint16_t numBearersPerUe = 1;
    double distance = 500.0;
    double yForUe = 500.0;
    double speed = 20.0;
    double simTime = (numberOfEnbs + 1) * distance / speed; // 75s
    double enbTxPowerDbm = 46.0;
    bool enableLteTraces = false;

    // ---- Defaults ----
    Config::SetDefault("ns3::UdpClient::Interval", TimeValue(MilliSeconds(10)));
    Config::SetDefault("ns3::UdpClient::MaxPackets", UintegerValue(1000000));
    Config::SetDefault("ns3::LteHelper::UseIdealRrc", BooleanValue(true));

    CommandLine cmd(__FILE__);
    cmd.AddValue("simTime", "Total duration of the simulation [s]", simTime);
    cmd.AddValue("speed", "UE speed [m/s]", speed);
    cmd.AddValue("distance", "Inter-eNB distance parameter [m]", distance);
    cmd.AddValue("yForUe", "UE y-coordinate [m]", yForUe);
    cmd.AddValue("enbTxPowerDbm", "eNB TX power [dBm]", enbTxPowerDbm);
    cmd.AddValue("enableLteTraces", "Enable LTE PHY/MAC/RLC/PDCP traces", enableLteTraces);
    cmd.Parse(argc, argv);

    // ---- LTE / EPC helpers ----
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);

    lteHelper->SetSchedulerType("ns3::RrFfMacScheduler");
    lteHelper->SetHandoverAlgorithmType("ns3::A2A4RsrqHandoverAlgorithm");
    lteHelper->SetHandoverAlgorithmAttribute("ServingCellThreshold", UintegerValue(30));
    lteHelper->SetHandoverAlgorithmAttribute("NeighbourCellOffset", UintegerValue(1));

    Ptr<Node> pgw = epcHelper->GetPgwNode();

    // ---- Remote host ----
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);

    InternetStackHelper internet;
    internet.Install(remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu", UintegerValue(1500));
    p2ph.SetChannelAttribute("Delay", TimeValue(Seconds(0.010)));

    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
    Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress(1);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"),
                                               Ipv4Mask("255.0.0.0"),
                                               1);

    NodeContainer ueNodes;
    NodeContainer enbNodes;
    enbNodes.Create(numberOfEnbs);
    ueNodes.Create(numberOfUes);

    Ptr<ListPositionAllocator> enbPositionAlloc = CreateObject<ListPositionAllocator>();
    for (uint16_t i = 0; i < numberOfEnbs; ++i)
    {
        enbPositionAlloc->Add(Vector(distance * (i + 1), distance, 0.0));
    }

    MobilityHelper enbMobility;
    enbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbMobility.SetPositionAllocator(enbPositionAlloc);
    enbMobility.Install(enbNodes);

    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    ueMobility.Install(ueNodes);

    ueNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(0.0, yForUe, 0.0));
    ueNodes.Get(0)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(
        Vector(speed, 0.0, 0.0));

    NodeContainer infraNodes;
    infraNodes.Add(pgw);
    infraNodes.Add(remoteHost);

    MobilityHelper infraMobility;
    infraMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    infraMobility.Install(infraNodes);

    infraNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(-50.0, distance, 0.0));   // PGW
    infraNodes.Get(1)->GetObject<MobilityModel>()->SetPosition(Vector(-200.0, distance, 0.0));  // RemoteHost

    Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(enbTxPowerDbm));
    NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice(ueNodes);

    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIpIfaces =
        epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLteDevs));

    for (uint16_t i = 0; i < numberOfUes; ++i)
    {
        lteHelper->Attach(ueLteDevs.Get(i), enbLteDevs.Get(0));
    }

    for (uint32_t u = 0; u < ueNodes.GetN(); ++u)
    {
        Ptr<Node> ue = ueNodes.Get(u);
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(ue->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    uint16_t dlPort = 10000;
    uint16_t ulPort = 20000;

    Ptr<UniformRandomVariable> startTimeSeconds = CreateObject<UniformRandomVariable>();
    startTimeSeconds->SetAttribute("Min", DoubleValue(0.001));
    startTimeSeconds->SetAttribute("Max", DoubleValue(0.010));

    for (uint32_t u = 0; u < numberOfUes; ++u)
    {
        Ptr<Node> ue = ueNodes.Get(u);

        for (uint32_t b = 0; b < numBearersPerUe; ++b)
        {
            ++dlPort;
            ++ulPort;

            ApplicationContainer clientApps;
            ApplicationContainer serverApps;

            UdpClientHelper dlClient(ueIpIfaces.GetAddress(u), dlPort);
            clientApps.Add(dlClient.Install(remoteHost));

            PacketSinkHelper dlSink("ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), dlPort));
            serverApps.Add(dlSink.Install(ue));

            UdpClientHelper ulClient(remoteHostAddr, ulPort);
            clientApps.Add(ulClient.Install(ue));

            PacketSinkHelper ulSink("ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), ulPort));
            serverApps.Add(ulSink.Install(remoteHost));

            Ptr<EpcTft> tft = Create<EpcTft>();

            EpcTft::PacketFilter dlpf;
            dlpf.localPortStart = dlPort;
            dlpf.localPortEnd = dlPort;
            tft->Add(dlpf);

            EpcTft::PacketFilter ulpf;
            ulpf.remotePortStart = ulPort;
            ulpf.remotePortEnd = ulPort;
            tft->Add(ulpf);

            EpsBearer bearer(EpsBearer::NGBR_VIDEO_TCP_DEFAULT);
            lteHelper->ActivateDedicatedEpsBearer(ueLteDevs.Get(u), bearer, tft);

            Time startTime = Seconds(startTimeSeconds->GetValue());
            Time stopTime = Seconds(simTime - 0.01);

            serverApps.Start(startTime);
            clientApps.Start(startTime);
            serverApps.Stop(stopTime);
            clientApps.Stop(stopTime);
        }
    }

    lteHelper->AddX2Interface(enbNodes);

    std::vector<uint32_t> autoPlacedCoreNodeIds;
    double extraX = -150.0;
    double extraY = distance + 80.0;

    for (auto it = NodeList::Begin(); it != NodeList::End(); ++it)
    {
        Ptr<Node> n = *it;
        if (!n->GetObject<MobilityModel>())
        {
            MobilityHelper mh;
            mh.SetMobilityModel("ns3::ConstantPositionMobilityModel");
            mh.Install(n);
            n->GetObject<MobilityModel>()->SetPosition(Vector(extraX, extraY, 0.0));
            autoPlacedCoreNodeIds.push_back(n->GetId());
            extraY += 80.0;
        }
    }

    AnimationInterface anim("lte-ho-netanim.xml");
    anim.SetMobilityPollInterval(MilliSeconds(100));

    anim.EnablePacketMetadata(false);

    anim.SetStartTime(Seconds(0.0));
    anim.SetStopTime(Seconds(simTime));

    uint32_t enbIcon  = anim.AddResource("./images/devices/antennatower_vl.png");
    uint32_t ueIcon   = anim.AddResource("./images/devices/robot.png");
    uint32_t coreIcon = anim.AddResource("./images/devices/lan-bus_vl.png");
    uint32_t srvIcon  = anim.AddResource("./images/devices/server_vl.png");

    anim.UpdateNodeDescription(remoteHost, "");
    anim.UpdateNodeDescription(pgw, "");
    anim.UpdateNodeDescription(enbNodes.Get(0), "enb1");
    anim.UpdateNodeDescription(enbNodes.Get(1), "");
    anim.UpdateNodeDescription(ueNodes.Get(0), "robot");

    for (uint32_t nodeId : autoPlacedCoreNodeIds)
    {
        anim.UpdateNodeDescription(nodeId, "");
    }

    anim.UpdateNodeImage(enbNodes.Get(0)->GetId(), enbIcon);
    anim.UpdateNodeImage(enbNodes.Get(1)->GetId(), enbIcon);
    anim.UpdateNodeImage(ueNodes.Get(0)->GetId(), ueIcon);
    anim.UpdateNodeImage(pgw->GetId(), coreIcon);
    anim.UpdateNodeImage(remoteHost->GetId(), srvIcon);

    for (uint32_t nodeId : autoPlacedCoreNodeIds)
    {
        if (nodeId != pgw->GetId())
        {
            anim.UpdateNodeImage(nodeId, coreIcon);
        }
    }

    anim.UpdateNodeColor(remoteHost, 180, 0, 180);
    anim.UpdateNodeColor(pgw, 255, 140, 0);
    anim.UpdateNodeColor(enbNodes.Get(0), 255, 0, 0);
    anim.UpdateNodeColor(enbNodes.Get(1), 0, 0, 255);
    anim.UpdateNodeColor(ueNodes.Get(0), 0, 170, 0);

    anim.UpdateNodeSize(remoteHost->GetId(), 50, 50);
    anim.UpdateNodeSize(pgw->GetId(), 50, 50);
    anim.UpdateNodeSize(enbNodes.Get(0)->GetId(), 80, 80);
    anim.UpdateNodeSize(enbNodes.Get(1)->GetId(), 80, 80);
    anim.UpdateNodeSize(ueNodes.Get(0)->GetId(), 50, 50);

    for (uint32_t nodeId : autoPlacedCoreNodeIds)
    {
        if (nodeId != pgw->GetId())
        {
            anim.UpdateNodeSize(nodeId, 7.0, 7.0);
        }
    }

    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/ConnectionEstablished",
                    MakeCallback(&NotifyConnectionEstablishedEnb));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/ConnectionEstablished",
                    MakeCallback(&NotifyConnectionEstablishedUe));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverStart",
                    MakeCallback(&NotifyHandoverStartEnb));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/HandoverStart",
                    MakeCallback(&NotifyHandoverStartUe));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
                    MakeCallback(&NotifyHandoverEndOkEnb));
    Config::Connect("/NodeList/*/DeviceList/*/LteUeRrc/HandoverEndOk",
                    MakeCallback(&NotifyHandoverEndOkUe));

    if (enableLteTraces)
    {
        lteHelper->EnablePhyTraces();
        lteHelper->EnableMacTraces();
        lteHelper->EnableRlcTraces();
        lteHelper->EnablePdcpTraces();
    }

    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    Simulator::Destroy();
    return 0;
}